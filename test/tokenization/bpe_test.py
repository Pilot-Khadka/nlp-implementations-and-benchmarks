def main():
    training_text = """Byte pair encoding is a simple form of data compression in which the most common pair of consecutive bytes of data is replaced with a byte that does not occur within that data.
    Byte pair encoding is used in natural language processing to manage the vocabulary of large language models.
    The technique was first described in 1994 by Philip Gage.
    Byte pair encoding is simple but effective."""

    training_text = training_text * 3
    vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    ids = list(training_text.encode("utf-8"))
    nums_bytes = bytes(ids)
    print("nums str:", nums_bytes)

    print("decoded:", nums_bytes.decode("utf-8"))


if __name__ == "__main__":
    main()
