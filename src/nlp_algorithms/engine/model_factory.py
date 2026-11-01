import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_algorithms.engine.registry import (
    MODEL_REGISTRY,
    get_from_registry,
    TASK_REGISTRY,
)
from nlp_algorithms.engine.embedding_factory import EmbeddingFactory
from nlp_algorithms.encoder_only.bert import BERTClassifier

logger = logging.getLogger(__name__)


class EmbeddingDropout(nn.Module):
    def __init__(self, embedding: nn.Embedding, p: float = 0.1):
        super().__init__()
        self.embedding = embedding
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return self.embedding(x)
        vocab_size = self.embedding.weight.size(0)
        row_mask = self.embedding.weight.new_empty(vocab_size, 1).bernoulli_(
            1 - self.p
        ) / (1 - self.p)
        masked_weight = self.embedding.weight * row_mask
        return F.embedding(
            x,
            masked_weight,
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        )


class WordDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask = x.new_empty(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)
        return x * mask


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding,
        encoder,
        tie_weights: bool = False,
        embedding_dropout: float = 0.0,
        word_dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = embedding
        self.embedding_dropout = EmbeddingDropout(
            embedding=embedding, p=embedding_dropout
        )
        self.word_dropout = WordDropout(word_dropout)
        # lstm, rnn, gru, etc
        self.encoder = encoder
        hidden_size = encoder.hidden_dim
        vocab_size = embedding.num_embeddings
        emb_dim = embedding.embedding_dim
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            if hidden_size != emb_dim:
                raise ValueError(
                    f"Cannot tie weights: hidden_size={hidden_size} "
                    f"!= embedding_dim={emb_dim}"
                )
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, *args, **kwargs):
        emb = self.word_dropout(self.embedding_dropout(input_ids))
        outputs, hidden = self.encoder(emb, *args, **kwargs)
        logits = self.lm_head(outputs)
        return logits, hidden


class ClassificationModel(nn.Module):
    def __init__(
        self,
        embedding,
        encoder,
        num_classes,
        embedding_dropout: float = 0.0,
        word_dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = embedding
        self.embedding_dropout = EmbeddingDropout(
            embedding=embedding, p=embedding_dropout
        )
        self.word_dropout = WordDropout(word_dropout)
        self.encoder = encoder

        bidirectional = getattr(encoder, "bidirectional", False)
        hidden_dim = encoder.hidden_dim * 2 if bidirectional else encoder.hidden_dim

        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.bidirectional = bidirectional

    def forward(self, input_ids):
        emb = self.word_dropout(self.embedding_dropout(input_ids))
        outputs, hidden = self.encoder(emb)

        # outputs shape: (B, T, H) or (B, T, 2H) if bidirectional
        logits = self.output_layer(outputs)
        return logits, hidden


def load_pretrained(model: nn.Module, path: str) -> None:
    """
    description: load a state dict saved by train.py into a model built for this
        config, not strict, so a model can start from one pretrained on another task
        (lm -> classification, bert mlm -> classification, more pretraining).
        the token ids have to be the same, reuse the vocab with tokenizer.path

    inputs:
        model: the model built from the config
        path: model_best.pt / model_final.pt of the earlier run
    """
    state = torch.load(path, map_location="cpu", weights_only=True)

    target, prefix = model, ""
    if not model.state_dict().keys() & state.keys():
        # a wrapper around the pretrained model, e.g. BERTClassifier(bert)
        for name, child in model.named_children():
            if child.state_dict().keys() & state.keys():
                target, prefix = child, f"{name}."
                break

    own = target.state_dict()
    # usually another vocab (tokenizer.path not set) or model size, not a silent skip
    mismatched = [
        f"{k} {tuple(state[k].shape)} vs {tuple(own[k].shape)}"
        for k in state
        if k in own and state[k].shape != own[k].shape
    ]
    if mismatched:
        raise ValueError(
            f"model.pretrained {path}: other shapes than the model, "
            f"{', '.join(mismatched[:4])}"
        )
    # by hand, load_state_dict does not report keys of a child set to None
    # (BERTClassifier drops the mlm and nsp heads that way)
    unexpected = [k for k in state if k not in own]
    loaded = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k not in loaded]
    if not loaded:
        raise ValueError(f"model.pretrained {path}: no parameter matches the model")

    target.load_state_dict(loaded, strict=False)
    logger.info(
        f"loaded {len(loaded)} pretrained tensors from {path} "
        f"into '{prefix.rstrip('.') or 'model'}'"
    )
    # missing = trained from scratch, unexpected = not used (e.g. the mlm head)
    for name, keys in [
        ("new (not in the checkpoint)", missing),
        ("unused", unexpected),
    ]:
        if keys:
            shown = ", ".join(sorted(keys)[:8])
            more = f" (+{len(keys) - 8} more)" if len(keys) > 8 else ""
            logger.info(f"  {name}: {shown}{more}")


class ModelFactory:
    @staticmethod
    def create_model(
        model_config,
        dataset_bundle,
        task_config,
        data_config,
    ):
        model = ModelFactory._build_model(
            model_config, dataset_bundle, task_config, data_config
        )
        # model.pretrained works for every model, e.g. bert_classification_imdb.yaml
        pretrained = model_config.get("pretrained")
        if pretrained:
            load_pretrained(model, pretrained)
        return model

    @staticmethod
    def _build_model(
        model_config,
        dataset_bundle,
        task_config,
        data_config,
    ):
        task = get_from_registry(TASK_REGISTRY, task_config.name)
        all_flags = [k for k, v in model_config.items() if isinstance(v, bool) and v]
        model_variants = MODEL_REGISTRY[model_config.name].get("variants", {})

        variant_flag_names = set()
        for variant_key in model_variants.keys():
            variant_flag_names.update(variant_key)

        model_flags = [f for f in all_flags if f in variant_flag_names]
        other_flags = [f for f in all_flags if f not in variant_flag_names]

        if hasattr(task, "allowed_flags"):
            allowed = getattr(task, "allowed_flags")
            invalid_flags = set(other_flags) - set(allowed)
            if invalid_flags:
                raise ValueError(
                    f"Flags {invalid_flags} are not allowed for task {task_config.name}. "
                    f"Allowed flags: {allowed}"
                )
        elif other_flags:
            raise ValueError(
                f"Task {task_config.name} does not support any flags, "
                f"but received: {other_flags}"
            )

        if len(model_variants) > 1 and not model_flags:
            model_entry = MODEL_REGISTRY[model_config.name]
            if "default" in model_entry:
                pass
            elif frozenset({"unidirectional"}) in model_variants:
                model_flags = ["unidirectional"]
            else:
                raise ValueError(
                    f"Model '{model_config.name}' has multiple variants {[set(k) for k in model_variants.keys()]} "
                    f"but no flag was specified. Please provide one of the variant flags."
                )

        model_class = get_from_registry(
            MODEL_REGISTRY, model_config.name, flags=model_flags
        )

        # pretrained is loaded by create_model, it is not a constructor argument
        not_kwargs = {"name", "pretrained"}

        if task_config.name == "masked_language_modeling":
            # bert has its own token, position and segment embeddings and heads
            kwargs = {k: v for k, v in model_config.items() if k not in not_kwargs}
            return model_class(vocab_size=len(dataset_bundle.vocab), **kwargs)

        if task_config.name == "language_modeling" and model_config.name == "gpt2":
            # gpt-2 has its own token and position embeddings and a tied lm head
            kwargs = {k: v for k, v in model_config.items() if k not in not_kwargs}
            return model_class(vocab_size=len(dataset_bundle.vocab), **kwargs)

        if task_config.name == "classification" and model_config.name == "bert":
            kwargs = {k: v for k, v in model_config.items() if k not in not_kwargs}
            bert = model_class(vocab_size=len(dataset_bundle.vocab), **kwargs)
            # model.pretrained (the mlm + nsp checkpoint) goes into the wrapped bert
            return BERTClassifier(
                bert, data_config.num_class, dropout=model_config.get("dropout", 0.1)
            )

        embedding = EmbeddingFactory.create(model_config, dataset_bundle.vocab)

        # Variant flags drive class selection in the registry, so they are not
        # forwarded as constructor kwargs, the selected class already encodes that
        # behaviour (e.g. a bidirectional LSTM class rather than passing bidirectional=True).
        explicit_kwargs = {"input_dim": embedding.embedding_dim}

        encoder_kwargs = {
            k: v
            for k, v in model_config.items()
            if k not in variant_flag_names
            and k not in explicit_kwargs
            and k != "pretrained"
        }

        encoder = model_class(**explicit_kwargs, **encoder_kwargs)

        weight_tying = model_config.get("weight_tying", False)
        embedding_dropout = model_config.get("embedding_dropout", 0.0)
        word_dropout = model_config.get("embedding_word_dropout", 0.0)
        if task_config.name == "language_modeling":
            return LanguageModel(
                embedding=embedding,
                encoder=encoder,
                tie_weights=weight_tying,
                embedding_dropout=embedding_dropout,
                word_dropout=word_dropout,
            )

        if task_config.name == "classification":
            num_classes = data_config.num_class
            return ClassificationModel(
                embedding=embedding,
                encoder=encoder,
                num_classes=num_classes,
                embedding_dropout=embedding_dropout,
                word_dropout=word_dropout,
            )

        raise ValueError(f"Unsupported task: {task_config.name}")
