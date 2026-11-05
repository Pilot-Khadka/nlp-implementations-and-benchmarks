import pytest
from collections import Counter
from nlp_algorithms.vectorization.bow import TextReader, BagOfWords
from nlp_algorithms.util.data import get_data_file, TINY_SHAKESPEARE_URL


@pytest.fixture
def shakespeare_corpus():
    return list(TextReader().read(get_data_file("tiny_shakespeare.txt", TINY_SHAKESPEARE_URL)))


def test_bow_shakespeare(shakespeare_corpus):
    assert len(shakespeare_corpus) > 0, "Corpus should not be empty"

    bow = BagOfWords()
    vectors = list(bow.fit_transform(shakespeare_corpus))

    # Number of vectors must equal number of lines
    assert len(vectors) == len(shakespeare_corpus)

    # Output vectors must be Counters
    assert all(isinstance(v, Counter) for v in vectors)

    # Vocabulary size <= max_vocab (50,000)
    assert len(bow.vocabulary) <= 50000

    # All vector indices must be valid vocabulary indices
    vocab_indices = set(bow.vocabulary.values())
    for vec in vectors:
        assert all(idx in vocab_indices for idx in vec.keys())

    bow2 = BagOfWords()
    manual = list(bow2.fit(shakespeare_corpus).transform(shakespeare_corpus))
    assert vectors == manual
    assert bow.vocabulary == bow2.vocabulary
