# nlp-implementations-and-benchmarks
This repo implements classic and modern nlp algorithms, from tokenization and RNNs to encoder-only, encoder-decoder and decoder-only transformers, and benchmarks them against reported scores and the PyTorch equivalents.

## results

Best custom model per task. Full tables and notes are in [model comparison](#model-comparison).

| Task                      | Dataset                    | Custom model                  | Score                   | Reference                          |
| ------------------------- | -------------------------- | ----------------------------- | ----------------------- | ---------------------------------- |
| Classification            | imdb                       | BERT (small, pretrained on imdb) | 89.65 test accuracy  | BERT-Base 94.60                    |
| Classification            | imdb                       | GRU (best RNN)                | 81.09 test accuracy     | PyTorch BiGRU 81.62                |
| Language modeling         | penn treebank              | AWD-LSTM                      | 86.42 test perplexity   | official AWD-LSTM 58.8             |
| Language modeling         | penn treebank              | GPT-2 (small)                 | 93.84 test perplexity   | GPT-2 117M zero-shot 65.85         |
| Masked language modeling  | imdb (75k reviews)         | BERT (small)                  | 18.9 MLM perplexity     | -                                  |
| Machine translation       | multi30k de->en            | Transformer-Base              | 23.91 BLEU (greedy)     | 26.4                               |
| Machine translation       | ne-en 208k, FLORES+        | Transformer                   | 13.30 BLEU (beam 5)     | 13.0 (Duwal & Bal 2019)            |

## setup

Requires python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Pilot-Khadka/nlp-algorithms.git
cd nlp-algorithms
uv sync
```

`uv sync` installs the `nlp_algorithms` package (from `src/`) in editable mode with torch and the other dependencies. A GPU is optional, everything runs on cpu, but the full runs in the tables were trained on kaggle T4s.

Datasets are downloaded on the first run into `data/` (penn treebank, tatoeba from their urls, imdb, multi30k and ne-en from huggingface). The ne->en test set (`openlanguagedata/flores_plus`) is gated on huggingface: accept its terms on the dataset page and log in first.

```bash
uv run huggingface-cli login
```

## running experiments

Every experiment is a yaml config. Train with `train.py`, then score the test set with `evaluate.py`:

```bash
uv run python train.py --path config/ptb/gpt2_ptb.yaml --verbose
uv run python evaluate.py --path config/ptb/gpt2_ptb.yaml
```

- `--verbose` shows progress bars, without it only the log lines are printed (also written to `log/training.log`).
- Checkpoints go to `train.checkpoint_dir`, or `checkpoint/<dataset>/<config name>` if it is not set. `model_best.pt` is the best epoch on the validation split, `model_final.pt` the last epoch.
- `evaluate.py` scores `model_best.pt` by default and writes `eval_model_best.json` next to it. Use `--checkpoint path/to/model.pt` for another checkpoint (e.g. `model_final.pt`).
- `train.multi_gpu: true` trains on all visible gpus with DDP, it falls back to one gpu or cpu when there are fewer.
- `dataset.max_samples: 500` gives a quick smoke run on a small subset.

### configs

| Config                                        | Task                      | What it runs                                             |
| --------------------------------------------- | ------------------------- | -------------------------------------------------------- |
| `config/imdb/{rnn,gru,lstm,birnn,bigru,bilstm}_imdb.yaml` | classification | custom RNN family on imdb sentiment             |
| `config/imdb/pytorch_*_imdb.yaml`             | classification            | the same models with PyTorch's `nn.RNN/GRU/LSTM`         |
| `config/imdb/{qrnn,awd_lstm}_imdb.yaml`       | classification            | QRNN and AWD-LSTM                                        |
| `config/imdb/bert_imdb.yaml`                  | masked_language_modeling  | BERT pretraining (MLM + NSP) on imdb train + unsupervised |
| `config/imdb/bert_classification_imdb.yaml`   | classification            | fine-tunes the pretrained BERT on imdb sentiment         |
| `config/ptb/{gru,pytorch_gru,qrnn,awd_lstm}_ptb.yaml` | language_modeling | RNN language models on penn treebank                   |
| `config/ptb/gpt2_ptb.yaml`                    | language_modeling         | small GPT-2 on penn treebank                             |
| `config/multi30k/transformer_multi30k.yaml`   | machine_translation       | Transformer-Base, multi30k de->en                         |
| `config/ne_en/transformer_ne_en.yaml`         | machine_translation       | Transformer, nepali->english (ne-en 208k, FLORES+)       |
| `config/seq2seq_tatoeba.yaml`                 | machine_translation       | Transformer on a tatoeba language pair (eng-nep by default) |

### bert: pretrain, then fine-tune

```bash
uv run python train.py --path config/imdb/bert_imdb.yaml
uv run python train.py --path config/imdb/bert_classification_imdb.yaml
uv run python evaluate.py --path config/imdb/bert_classification_imdb.yaml
```

The fine-tuning config loads the pretrained weights from `model.pretrained` and the pretraining tokenizer from `tokenizer.path`. Point both to the files written by the pretraining run.

### reusing weights

`model.pretrained: path/to/model.pt` works for every model built from a config. Weights are matched by name and the rest stays freshly initialized (e.g. a classification head), both lists are logged. A shape mismatch (another vocab or model size) is an error. This also allows continuing BERT pretraining, RNN language model -> RNN classifier, or GPT-2 -> GPT-2.

### machine translation

Translation configs use their own pipeline (joint or separate byte-level BPE, token-based batches, label smoothing, warmup schedule). `evaluate.py` reports sacreBLEU with greedy and beam search decoding, and also scores the average of the last `train.keep_last` checkpoints. Beam size and length penalty are set in `task.beam_size` and `task.len_penalty`.

### standalone modules

The classic algorithms have small demos that run on their own. The data they need is downloaded into `data/` on the first run (tiny shakespeare, names, WordNet 3.1, the 1 billion word benchmark for word2vec).

```bash
uv run python -m nlp_algorithms.tokenization.bpe                      # byte-level BPE on tiny shakespeare
uv run python -m nlp_algorithms.stemming.porter_stemmer
uv run python -m nlp_algorithms.lemmatization.lemmatization           # rule based
uv run python -m nlp_algorithms.lemmatization.lemmatization_wordnet   # rules + WordNet lookup
uv run python -m nlp_algorithms.vectorization.tfidf
uv run python -m nlp_algorithms.vectorization.word2vec.skipgram       # skip-gram pairs with negative sampling
uv run python -m nlp_algorithms.vectorization.word2vec.train --max_sentences 100000   # 1.7 GB download
uv run python -m nlp_algorithms.lm_ngram_lm.mlp                       # character level MLP language model on names
uv run pytest test/
```

- The porter stemmer matches NLTK's original-algorithm porter stemmer on 99.97% of the 22,987 distinct words in the complete works of Shakespeare (the rest are 1-2 letter words, which the reference implementation leaves unchanged).
- The character level MLP (block size 3, 10-d embeddings, 200 hidden) reaches 2.14 train / 2.18 valid / 2.15 test loss on names.txt after 200k steps.

## project structure

```
config/                      yaml configs, one per experiment
train.py, evaluate.py        entry points
src/nlp_algorithms/
  tokenization/              whitespace, byte-level BPE
  stemming/, lemmatization/  porter stemmer, wordnet lemmatizer
  vectorization/             one-hot, n-gram, bag of words, tf-idf, word2vec
  rnn/, gru/, lstm/          RNN family, QRNN, AWD-LSTM (custom and PyTorch versions)
  attention/                 multi-head attention (causal for decoder-only)
  encoder_only/              BERT
  encoder_decoder/           seq2seq transformer (RoPE), training, beam search, BLEU
  decoder_only/              GPT-2
  dataset/                   downloaders and readers (ptb, imdb, tatoeba, huggingface)
  engine/                    registry, model / dataset / optimizer factories
  task/                      per-task preprocessing and metrics
  training/                  trainers per task, train / eval orchestration, DDP
test/                        tokenization and vectorization tests
```

## components:
- [x] Tokenization (whitespace, BPE)
- [x] Stemming and Lemmatization
- [x] Vectorization (One-hot, N-gram, TF-IDF, Word2vec)
- [x] RNN-family architectures
- [x] Transformer models
  - [x] Sequence-to-Sequence
  - [x] Encoder only (BERT)
  - [x] Decoder only (GPT-2)
- [x] NLP tasks
  - [x] Classification
  - [x] Language Modeling
  - [x] Machine Translation

## architectures

| Architecture                | Status        |
| --------------------------- | ------------- |
| RNN (vanilla, stacked)      | Implemented   |
| GRU / Bi-GRU                | Implemented   |
| LSTM / Bi-LSTM              | Implemented   |
| AWD-LSTM                    | Implemented   |
| Transformer Attention       | Implemented   |
| Transformer Encoder only model (BERT) | Implemented   |
| Transformer Encoder-Decoder | Implemented   |
| Transformer Decoder-only model (GPT-2) | Implemented   |

## datasets

- [x] Penn treebank (language modeling)
- [x] Imdb (classification)
- [x] Tatoeba (machine translation)
- [x] Huggingface datasets

## model comparison
Comparison between custom model implementation and PyTorch equivalents

### classification (imdb dataset)

| Model      | Implementation | Train Loss | Train Time (avg) | Valid Loss | Valid Time (avg) | Accuracy | Precision | Recall | F1     |
| -----------|----------------| ---------- | ---------------- | ---------- | ---------------- | -------- | --------- | ------ | ------ |
| **BiGRU**  |Pytorch         | 0.0143     | 34.39s           | 0.9941     | 11.77s           | 0.8162   | 0.8172    | 0.8162 | 0.8167 |
| **BiGRU**  |Custom          | 0.0125     | 382.14s          | 1.3821     | 111.09s          | 0.8086   | 0.8102    | 0.8086 | 0.8094 |
| **GRU**    |Pytorch         | 0.0154     | 16.70s           | 1.3159     | 5.40s            | 0.7993   | 0.8028    | 0.7993 | 0.8010 |
| **GRU**    |Custom          | 0.0175     | 157.92s          | 1.1263     | 49.58s           | 0.8109   | 0.8109    | 0.8109 | 0.8109 |
| **BiLSTM** |Pytorch         | 0.0239     | 40.11s           | 1.3207     | 14.41s           | 0.7825   | 0.7840    | 0.7825 | 0.7832 |
| **BiLSTM** |Custom          | 0.0203     | 365.49s          | 1.1990     | 88.54s           | 0.7948   | 0.7951    | 0.7948 | 0.7949 |
| **LSTM**   |Pytorch         | 0.0540     | 19.41s           | 1.1087     | 7.28s            | 0.7839   | 0.7840    | 0.7839 | 0.7840 |
| **LSTM**   |Custom          | 0.0266     | 176.13s          | 1.5475     | 48.30s           | 0.7376   | 0.7377    | 0.7376 | 0.7377 |
| **QRNN**     | Custom         | 0.0106     | 110.53s          | 1.7951     | 15.72s           | 0.7281   | 0.7290    | 0.7281 | 0.7286 |
| **AWD-LSTM** | Custom         | 0.0806     | 292.92s          | 0.8338     | 45.44s           | 0.8020   | 0.8028    | 0.8020 | 0.8024 |

**Notes**

- **PyTorch implementations are ~8-10x faster** due to optimized CUDA kernels and fused operations.
- RNN / BiRNN models for both custom and pytorch were excluded since they converged to ~50% accuracy (random).

### bert (imdb dataset)

Pretraining (masked language modeling + next sentence prediction) on the imdb reviews, then fine-tuning on imdb sentiment classification.

| Stage       | Config                                                                                           | Data                                   | MLM Loss | MLM Accuracy | MLM Perplexity | NSP Accuracy | Train Time |
| ----------- | ------------------------------------------------------------------------------------------------ | -------------------------------------- | -------- | ------------ | -------------- | ------------ | ---------- |
| Pretraining | 4 layers, d=256, FF=1024, 4 heads, 5.85M params, byte-level BPE 10k, SeqLen=256, BS=128, 150 epochs | imdb train + unsupervised (75k reviews) | 2.94     | 0.446        | 18.9           | 0.891        | 2h36m (2x T4) |

| Model                  | Implementation                                                                                     | Config (Key Hyperparams)                                          | Pretraining Data                          | Test Accuracy |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------- | ------------- |
| **BERT-Large**         | [reported score from Yang et al. 2019 (XLNet), Table 4](https://arxiv.org/abs/1906.08237)          | 24 layers, d=1024, 340M params                                    | BooksCorpus + Wikipedia                   | 95.49         |
| **BERT-Base**          | [reported score from Sun et al. 2019, Table 5](https://arxiv.org/abs/1905.05583)                   | 12 layers, d=768, 110M params, SeqLen=512 (head+tail), lr 2e-5    | BooksCorpus + Wikipedia                   | 94.60         |
| **BERT-Base**          | [reported score from Sun et al. 2019, Table 5](https://arxiv.org/abs/1905.05583)                   | same, further pretrained on the imdb train set for 100k steps     | BooksCorpus + Wikipedia, then imdb        | **95.63**     |
| **BERT (small)**       | Custom                                                                                             | 4 layers, d=256, 5.85M params, SeqLen=256 (head only), lr 1e-4, 4 epochs | imdb only (75k reviews)            | **89.65** |

**Notes**

- The reported scores are given as test error rates in the papers (4.51, 5.40 and 4.37), shown here as accuracy.
- The custom model is ~19x smaller than BERT-Base, sees only the first 254 tokens of a review (imdb reviews average ~300 word pieces) and is pretrained on ~100M characters of movie reviews instead of 3.3B words.
- There is no imdb validation split. The custom score is the final model after 4 epochs (89.83 at epoch 2, the best epoch picked on the test set). Fine-tuning takes 2.6 min on 2x T4.
- With the same imdb data the small pretrained BERT beats the best RNN above (BiGRU, 81.62) by 8 points.

### language modeling (penn treebank)

| Model            | Implementation               | Config                                                                               | Epoch     | Test Perplexity | Train Time / Epoch |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------ | --------- | --------------- | ------------------ |
| **AWD-LSTM**     | Official (no finetuning)     | BS=20, SeqLen=70, Dropouti=0.4, Dropouth=0.25                                        | up to 500 | **58.8**        | 65s                |
| **AWD-LSTM**     | Custom                       | BS=32, SeqLen=70, Dropouti=0.4, Dropouth=0.25, 3-layer, 1000 hidden                  | 60        | 86.42           | 156s               |
| **QRNN**         | Official (no finetuning)     | BS=20, SeqLen=70, Dropouti=0.4, Dropouth=0.3, 4-layer, 1550 hidden, emsize=400       | up to 550 | 58.3            | 28s                |
| **QRNN**         | Custom                       | BS=32, SeqLen=70, Dropouti=0.4, Dropouth=0.25, 3-layer, 1000 hidden, locked dropout | 60        | 113.21          | 118.69s            |
| **GRU**          | Custom (locked drop, embed drop, variational drop) | BS=16, SeqLen=70, Dropouti=0.4, EmbDrop=0.1, Dropouth=0.25, Dropout=0.4, 3-layer, 1000 hidden | 60 | **91.55** | 296s |
| **GRU**          | PyTorch (weight drop, locked drop, embed drop)     | BS=16, SeqLen=70, Dropouti=0.4, EmbDrop=0.1, Dropouth=0.25, Dropout=0.4, 3-layer, 1000 hidden | 60 | **91.61** | 101s |
| **GPT-2**        | [reported score from Radford et al. 2019, Table 3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 12 layers, d=768, 117M params, zero-shot (trained on WebText, not on ptb) | - | 65.85 | - |
| **GPT-2 (small)** | Custom                      | BS=32, SeqLen=128, random chunk offset, 4 layers, d=256, FF=1024, 4 heads, 5.78M params, dropout 0.4, attn/embed dropout 0.1, AdamW 6e-4 wd 0.1, warmup 200 + linear decay | 100 | 93.84 | 5.3s (T4) |

**Notes**

- The custom GPT-2 sees non-overlapping 128 token chunks with no state carried over (the RNNs carry their hidden state across batches). The chunk boundaries move to a random offset every epoch, so the chunks differ between epochs. Training takes 9.4 min on one T4.
- The best checkpoint is picked on ptb.valid (epoch 98, valid perplexity 102.18) and scored once on ptb.test. The last epoch (no checkpoint selection) scores 93.98 on test.
- The settings come from a sweep of 10 runs (60 epochs, picked on valid). Random chunk offsets gave the largest gain (106.97 -> 97.06 test), then more dropout and 128 instead of 256 token chunks (95.60), then 100 instead of 60 epochs (93.84). Larger models (d=384 and d=512, 6 layers) and more dropout (0.5+) were not better. The first run without these (256 token chunks, dropout 0.3, 40 epochs, checkpoint picked on test) got 109.27.
- PTB has ~930k training tokens, the train/valid gap keeps growing (train loss 3.90 vs valid loss 4.63 at the end) and the transformer does not beat the regularized RNNs. The reported GPT-2 score is zero-shot from a much larger model and dataset with a different tokenization, so it is not directly comparable.

### machine translation (multi30k de->en)

| Model                | Implementation               | Config (Key Hyperparams)                                              | Dataset        | Test BLEU |
| -------------------- | ---------------------------- | --------------------------------------------------------------------- | -------------- | --------- |
| **Transformer-Base** | [reported score from hyunwoongko/transformer](https://github.com/hyunwoongko/transformer) | 6L/6L, d=512, FF=2048, 8 heads, dropout=0.1, sinusoidal PE, BPE       | Multi30k En–De | **26.4**  |
| **Transformer-Base** | Custom                | 6L/6L, d=512, FF=2048, 8 heads, dropout=0.1, **RoPE**, byte-level BPE | Multi30k En–De | **23.91** (greedy decoding) |

* Reported score uses the its repository’s custom BLEU calculation.
* This repos custom implementation uses **sacreBLEU** for BLEU calculation.

### machine translation (nepali->english)

| Model                | Implementation               | Config (Key Hyperparams)                                              | Dataset        | Test BLEU |
| -------------------- | ---------------------------- | --------------------------------------------------------------------- | -------------- | --------- |
| **Transformer**      | [reported score from Duwal & Bal 2019](https://huggingface.co/datasets/sharad461/ne-en-parallel-208k) (fairseq) | 5L/5L, d=512, FF=2048, 8 heads, dropout=0.3, sinusoidal PE, joint SentencePiece BPE 2500, beam 5 | ne-en 208k, FLORES v1 devtest | **13.0** |
| **Transformer**      | Custom                | 5L/5L, d=512, FF=2048, 8 heads, dropout=0.3, **RoPE**, joint byte-level BPE 4000, shared embeddings, ~16k tokens/batch, 100 epochs | ne-en 208k, FLORES+ devtest | 11.16 (greedy decoding) |
| **Transformer**      | Custom                | same, last 5 checkpoints averaged, beam 5                             | ne-en 208k, FLORES+ devtest | **13.30** |

* Both are trained on the same 208k parallel sentences (no back-translated data).
* The reported score is on FLORES v1 devtest, this repo uses FLORES+ devtest (different sentences), so the scores are close but not the same test set.
* Nepali text is normalized (Devanagari digits to ASCII, danda spacing, zero-width characters) and duplicate pairs are removed before training.
