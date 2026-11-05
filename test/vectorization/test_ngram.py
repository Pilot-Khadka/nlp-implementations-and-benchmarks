import pytest
from collections import Counter

from nlp_algorithms.vectorization.ngram import NgramGen, NgramBoW

DOCS = [
    "hello there how are you",
    "I hope you are doing well",
    "there is more sand grains than stars in the universe",
]


class TestNgramGen:
    def test_bigrams_yield_correct_pairs(self):
        gen = NgramGen(2)
        result = list(gen.generate("hello there how"))
        assert result == ["hello there", "there how"]

    def test_trigrams_span_three_words(self):
        gen = NgramGen(3)
        result = list(gen.generate("a b c d"))
        assert result == ["a b c", "b c d"]

    def test_n_equals_text_length_yields_one(self):
        gen = NgramGen(3)
        assert list(gen.generate("a b c")) == ["a b c"]

    def test_n_exceeds_text_length_yields_nothing(self):
        gen = NgramGen(5)
        assert list(gen.generate("only three words")) == []

    def test_single_word_unigram(self):
        gen = NgramGen(1)
        assert list(gen.generate("hello")) == ["hello"]


class TestNgramBoW:
    def test_fit_builds_vocab_from_corpus(self):
        model = NgramBoW(n=2).fit(DOCS)
        assert len(model.vocab) > 0
        assert all(isinstance(k, str) for k in model.vocab)

    def test_vocab_respects_max_vocab(self):
        model = NgramBoW(n=1, max_vocab=3).fit(DOCS)
        assert len(model.vocab) <= 3

    def test_vocab_contains_most_frequent_ngrams(self):
        # "you are" appears in both doc 0 and doc 1, so must survive max_vocab=1
        model = NgramBoW(n=2, max_vocab=1).fit(["you are here", "you are there"])
        assert "you are" in model.vocab

    def test_fit_returns_self_for_chaining(self):
        model = NgramBoW(n=1)
        assert model.fit(DOCS) is model

    def test_transform_yields_one_counter_per_doc(self):
        model = NgramBoW(n=2).fit(DOCS)
        results = list(model.transform(DOCS))
        assert len(results) == len(DOCS)
        assert all(isinstance(r, Counter) for r in results)

    def test_transform_skips_oov_ngrams(self):
        model = NgramBoW(n=1, max_vocab=2).fit(DOCS)
        for vec in model.transform(DOCS):
            assert all(idx < 2 for idx in vec)

    def test_transform_counts_are_non_negative(self):
        model = NgramBoW(n=2).fit(DOCS)
        for vec in model.transform(DOCS):
            assert all(count >= 0 for count in vec.values())

    def test_fit_transform_matches_separate_calls(self):
        m1 = NgramBoW(n=2)
        combined = list(m1.fit_transform(DOCS))

        m2 = NgramBoW(n=2)
        m2.fit(DOCS)
        separate = list(m2.transform(DOCS))

        assert combined == separate

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_unigram_bigram_trigram_produce_results(self, n):
        model = NgramBoW(n=n).fit(DOCS)
        results = list(model.transform(DOCS))
        assert len(results) == len(DOCS)
