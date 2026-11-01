import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_algorithms.attention.attention import MultiHeadSelfAttention
from nlp_algorithms.engine.registry import register_model


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # gpt-2 uses gelu (tanh approximation) instead of relu
        return self.linear2(F.gelu(self.linear1(x), approximate="tanh"))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, attn_dropout)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # pre-norm as in gpt-2, a token only attends to itself and the tokens before it
        attn_out = self.attention(self.norm1(x), is_causal=True)
        x = x + self.dropout1(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_out)
        return x


@register_model("gpt2")
class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len=1024,
        dropout=0.1,
        attn_dropout=0.0,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # drops whole words from the vocabulary (dropoute in awd-lstm), for small data
        self.embedding_dropout = embedding_dropout
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout, attn_dropout)
                for _ in range(num_layers)
            ]
        )
        # the layers are pre-norm, so the last output is not normalized yet
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        # the residual projections are scaled by 1/sqrt(2 * num_layers), every layer
        # adds two of them (attention + feed forward) to the residual stream
        for name, param in self.named_parameters():
            if name.endswith(("attention.out_proj.weight", "ff.linear2.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
        # the lm head shares weights with the token embedding
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        # gpt-2 init, N(0, 0.02) instead of the default N(0, 1) embeddings
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _embed(self, input_ids):
        p = self.embedding_dropout
        if not self.training or p == 0:
            return self.token_embedding(input_ids)
        weight = self.token_embedding.weight
        # one keep / drop per word type, the kept rows are scaled by 1 / (1 - p)
        row_mask = weight.new_empty(weight.size(0), 1).bernoulli_(1 - p) / (1 - p)
        return F.embedding(input_ids, weight * row_mask)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} > max_len {self.max_len}")

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        hidden_states = self._embed(input_ids) + self.position_embedding(position_ids)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # (batch, seq, vocab), logits at t predict the token at t + 1
        return self.lm_head(self.final_norm(hidden_states))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        description: sample max_new_tokens tokens after input_ids, one at a time.
            temperature 0 is greedy decoding, top_k keeps only the k most likely tokens.

        inputs:
            input_ids: (batch, seq) prompt token ids
        """
        for _ in range(max_new_tokens):
            # no kv cache, the context is cut to the last max_len tokens
            context = input_ids[:, -self.max_len :]
            logits = self(context)[:, -1, :]

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


if __name__ == "__main__":
    pass
