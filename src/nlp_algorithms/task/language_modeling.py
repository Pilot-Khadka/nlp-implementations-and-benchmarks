import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTask
from nlp_algorithms.engine.registry import register_task
from nlp_algorithms.util.metric import perplexity


@register_task("language_modeling")
class LMTask(BaseTask):
    allowed_flags = {"pytorch", "causal", "weight_tying", "use_locked_dropout"}

    @staticmethod
    def get_output_dim(dataset_bundle):
        return len(dataset_bundle.vocab)

    @staticmethod
    def get_loss(pad_idx=0):
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

    @staticmethod
    def _logits(outputs):
        # the rnn models return (logits, hidden), gpt-2 only the logits
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def postprocess(self, outputs):
        # the most likely next token at every position
        return self._logits(outputs).argmax(dim=-1)

    def compute_metrics(self, outputs, batch):
        logits = self._logits(outputs).float()
        _, targets = batch
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0
        )
        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        return {"perplexity": perplexity(loss=loss), "accuracy": accuracy}
