class BytePairEncoder:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # 256 single bytes are base vocab
        self.num_merges = vocab_size - 256

        self.merges = {}  # (token_id1, token_id2) -> new_token_id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    def get_stats(self, ids):
        """Count frequency of adjacent pairs"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, new_id):
        """Replace all occurrences of pair with new_id"""
        new_ids = []
        i = 0
        while i < len(ids):
            # Check if we can merge at current position
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2  # Skip both tokens
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        ids = list(text.encode("utf-8"))
        print(f"Initial sequence length: {len(ids)} tokens")

        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break

            # most frequent pair
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]

            if best_count < 2:
                print("Stopping early: no pairs with frequency >= 2")
                break

            # new token for most freq pair
            new_id = 256 + i
            self.merges[best_pair] = new_id

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids = self.merge(ids, best_pair, new_id)

            print(
                f"Merge {i + 1}: {best_pair} -> {new_id} "
                f"(freq={best_count}, new_len={len(ids)})"
            )

        print(f"Final sequence length: {len(ids)} tokens")
        print(f"Compression ratio: {len(text.encode('utf-8')) / len(ids):.2f}x")
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            best_pair = min(
                zip(ids, ids[1:]),
                key=lambda p: self.merges.get(p, float("inf")),
            )

            if best_pair not in self.merges:
                break

            ids = self.merge(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids):
        bytes_list = b"".join(self.vocab[id] for id in ids)
        return bytes_list.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Sample training data with repetitive patterns
    text = """The Dark Knight Rises is a superhero movie released in 2012. It is the final part of Christopher Nolan’s Dark Knight trilogy, following Batman Begins and The Dark Knight. The film stars Christian Bale as Bruce Wayne/Batman, who has been retired as Batman for eight years after the events of the previous movie.

The main villain in the movie is Bane, played by Tom Hardy. Bane is a powerful and intelligent terrorist who threatens Gotham City with destruction. He forces Bruce Wayne to come out of retirement and become Batman again. Anne Hathaway plays Selina Kyle, also known as Catwoman, a skilled thief with her own agenda.

The movie is about Bruce Wayne’s struggle to overcome his physical and emotional challenges to save Gotham. It also shows themes of hope, sacrifice, and resilience. The film has many exciting action scenes, such as a plane hijack and a massive battle in Gotham.

In the end, Batman saves the city and inspires the people of Gotham. However, he is believed to have sacrificed his life. The movie ends with a twist, suggesting that Bruce Wayne is alive and has moved on to live a quiet life.

The Dark Knight Rises was a big success and is loved by many fans for its epic story, strong characters, and thrilling action.

"""

    bpe = BytePairEncoder(vocab_size=280)  # 256 base + 24 merges
    bpe.train(text)

    # Show learned vocabulary
    print(f"\n--- Learned Vocabulary (size: {len(bpe.vocab)}) ---")
    for i in range(256, len(bpe.vocab)):
        token_bytes = bpe.vocab[i]
        try:
            display = token_bytes.decode("utf-8")
        except:
            display = f"<bytes: {token_bytes.hex()}>"
        print(f"  ID {i}: {display!r}")

    # Test encoding/decoding
    test_text = "Byte pair encoding is simple"
    print(f"\n--- Encoding Test ---")
    print(f"Input: {test_text!r}")

    encoded = bpe.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = bpe.decode(encoded)
    print(f"Decoded: {decoded!r}")
    print(f"Match: {test_text == decoded}")

    # Compare compression
    original_bytes = len(test_text.encode("utf-8"))
    encoded_tokens = len(encoded)
    print(f"\nOriginal bytes: {original_bytes}")
    print(f"Encoded tokens: {encoded_tokens}")
    print(f"Compression: {original_bytes / encoded_tokens:.2f}x")

    # Show merge rules
    print(f"\n--- Merge Rules Learned ({len(bpe.merges)} total) ---")
    for i, ((a, b), new_id) in enumerate(list(bpe.merges.items())[:10]):
        a_str = bpe.vocab[a].decode("utf-8", errors="replace")
        b_str = bpe.vocab[b].decode("utf-8", errors="replace")
        new_str = bpe.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {i + 1}. ({a_str!r}, {b_str!r}) -> {new_id} ({new_str!r})")
