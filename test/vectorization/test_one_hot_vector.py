import pytest
from nlp_algorithms.vectorization.one_hot_encoding import OneHotEncoder

DOCS = [
    "hello there how are you",
    "I hope you are doing well",
    "there is more sand grains than stars in the universe",
]


class TestFit:
    def test_vocab_keys_are_lowercased(self):
        model = OneHotEncoder().fit(["Hello WORLD"])
        assert "hello" in model.vocab
        assert "world" in model.vocab
        assert "Hello" not in model.vocab

    def test_vocab_indices_are_unique(self):
        model = OneHotEncoder().fit(DOCS)
        indices = list(model.vocab.values())
        assert len(indices) == len(set(indices))

    def test_most_frequent_word_gets_index_zero(self):
        # "are" appears in two docs, so it should rank first
        model = OneHotEncoder().fit(["are are are", "are once"])
        assert model.vocab["are"] == 0

    def test_max_vocab_caps_vocabulary_size(self):
        model = OneHotEncoder(max_vocab=3).fit(DOCS)
        assert len(model.vocab) <= 3

    def test_fit_returns_self(self):
        model = OneHotEncoder()
        assert model.fit(DOCS) is model

    def test_fit_accepts_generator(self):
        model = OneHotEncoder().fit(d for d in DOCS)
        assert len(model.vocab) > 0


class TestTransform:
    def test_yields_one_list_per_document(self):
        model = OneHotEncoder().fit(DOCS)
        result = list(model.transform(DOCS))
        assert len(result) == len(DOCS)

    def test_each_word_gets_one_vector(self):
        model = OneHotEncoder().fit(DOCS)
        result = list(model.transform(["hello there how"]))
        assert len(result[0]) == 3

    def test_vector_length_equals_vocab_size(self):
        model = OneHotEncoder().fit(DOCS)
        vocab_size = len(model.vocab)
        for doc_vectors in model.transform(DOCS):
            assert all(len(v) == vocab_size for v in doc_vectors)

    def test_exactly_one_hot_bit_set_for_known_word(self):
        model = OneHotEncoder().fit(DOCS)
        vectors = list(model.transform(["hello"]))[0]
        assert sum(vectors[0]) == 1

    def test_all_zeros_for_oov_word(self):
        model = OneHotEncoder(max_vocab=2).fit(["common words only"])
        vectors = list(model.transform(["xyzzy"]))[0]
        assert sum(vectors[0]) == 0

    def test_hot_bit_position_matches_vocab_index(self):
        model = OneHotEncoder().fit(["hello world"])
        idx = model.vocab["hello"]
        vector = list(model.transform(["hello"]))[0][0]
        assert vector[idx] == 1


class TestTransformSentence:
    def test_yields_one_vector_per_document(self):
        model = OneHotEncoder().fit(DOCS)
        result = list(model.transform_sentence(DOCS))
        assert len(result) == len(DOCS)

    def test_vector_length_equals_vocab_size(self):
        model = OneHotEncoder().fit(DOCS)
        vocab_size = len(model.vocab)
        for vec in model.transform_sentence(DOCS):
            assert len(vec) == vocab_size

    def test_repeated_word_still_sets_bit_once(self):
        model = OneHotEncoder().fit(["hello"])
        vec = list(model.transform_sentence(["hello hello hello"]))[0]
        assert vec[model.vocab["hello"]] == 1
        assert sum(vec) == 1

    def test_two_distinct_words_set_two_bits(self):
        model = OneHotEncoder().fit(["hello world"])
        vec = list(model.transform_sentence(["hello world"]))[0]
        assert sum(vec) == 2


class TestFitTransform:
    def test_matches_separate_fit_and_transform(self):
        m1 = OneHotEncoder()
        combined = list(m1.fit_transform(DOCS))

        m2 = OneHotEncoder()
        m2.fit(DOCS)
        separate = list(m2.transform(DOCS))

        assert combined == separate

    def test_accepts_generator_without_exhaustion(self):
        model = OneHotEncoder()
        result = list(model.fit_transform(d for d in DOCS))
        assert len(result) == len(DOCS)
