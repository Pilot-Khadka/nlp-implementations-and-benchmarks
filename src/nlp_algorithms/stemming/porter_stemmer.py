import re


class PorterStemmer:
    """
    References:
    original link:  https://tartarus.org/martin/PorterStemmer/
    algorithm:      https://vijinimallawaarachchi.com/2017/05/09/porter-stemming-algorithm/

    simple rule based stemmer, cruder than lemmatization.
    tends to produce stems that are not actual words
    """

    def __init__(self, word):
        self.word = word.lower()

    def is_consonant(self, word, i):
        # y is a vowel after a consonant (sky, happy), a consonant otherwise (yes, toy)
        if word[i] in "aeiou":
            return False
        if word[i] == "y":
            return i == 0 or not self.is_consonant(word, i - 1)
        return True

    def m(self, word=None):
        # number of vowel-consonant sequences, [C](VC)^m[V]
        word = self.word if word is None else word
        pattern = "".join("c" if self.is_consonant(word, i) else "v" for i in range(len(word)))
        return len(re.findall(r"v+c+", pattern))

    def contains_vowel(self, word=None):
        word = self.word if word is None else word
        return any(not self.is_consonant(word, i) for i in range(len(word)))

    def double_consonant(self, word=None):
        word = self.word if word is None else word
        if len(word) < 2:
            return False
        return word[-1] == word[-2] and self.is_consonant(word, len(word) - 1)

    def cvc(self, word=None):
        # *o: stem ends consonant-vowel-consonant, the last one not w, x or y (hop, not snow)
        word = self.word if word is None else word
        if len(word) < 3:
            return False
        n = len(word)
        return (
            self.is_consonant(word, n - 1)
            and not self.is_consonant(word, n - 2)
            and self.is_consonant(word, n - 3)
            and word[-1] not in "wxy"
        )

    def replace_suffix(self, suffix_map, min_m):
        # only the longest matching suffix is tried, its stem needs m > min_m
        for suffix in sorted(suffix_map, key=len, reverse=True):
            if self.word.endswith(suffix):
                stem = self.word[: -len(suffix)]
                if self.m(stem) > min_m:
                    self.word = stem + suffix_map[suffix]
                return

    def step1a(self):
        if self.word.endswith("sses"):
            self.word = self.word[:-2]
        elif self.word.endswith("ies"):
            self.word = self.word[:-2]
        elif self.word.endswith("s") and not self.word.endswith("ss"):
            self.word = self.word[:-1]

    def step1b(self):
        if self.word.endswith("eed"):
            if self.m(self.word[:-3]) > 0:
                self.word = self.word[:-1]
        elif self.word.endswith("ed"):
            if self.contains_vowel(self.word[:-2]):
                self.word = self.word[:-2]
                # only when ed was removed
                self.step1b_2()
        elif self.word.endswith("ing"):
            if self.contains_vowel(self.word[:-3]):
                self.word = self.word[:-3]
                self.step1b_2()

    def step1b_2(self):
        if self.word.endswith(("at", "bl", "iz")):
            self.word += "e"
        elif self.word[-1] not in "lsz" and self.double_consonant():
            # *d -> string ends with double consonant
            self.word = self.word[:-1]
        elif self.m() == 1 and self.cvc():
            # *o -> stem ends in cvc
            self.word += "e"

    def step1c(self):
        if self.word.endswith("y") and self.contains_vowel(self.word[:-1]):
            self.word = self.word[:-1] + "i"

    def step2(self):
        suffix_map = {
            "ational": "ate",
            "tional": "tion",
            "enci": "ence",
            "anci": "ance",
            "izer": "ize",
            "abli": "able",
            "alli": "al",
            "entli": "ent",
            "eli": "e",
            "ousli": "ous",
            "ization": "ize",
            "ation": "ate",
            "ator": "ate",
            "alism": "al",
            "iveness": "ive",
            "fulness": "ful",
            "ousness": "ous",
            "aliti": "al",
            "iviti": "ive",
            "biliti": "ble",
        }
        self.replace_suffix(suffix_map, min_m=0)

    def step3(self):
        suffix_map = {
            "icate": "ic",
            "ative": "",
            "alize": "al",
            "iciti": "ic",
            "ical": "ic",
            "ful": "",
            "ness": "",
        }
        self.replace_suffix(suffix_map, min_m=0)

    def step4(self):
        suffix_map = [
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ion",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        ]
        for suffix in sorted(suffix_map, key=len, reverse=True):
            if self.word.endswith(suffix):
                stem = self.word[: -len(suffix)]
                # ion only after s or t (adoption -> adopt, not onion -> on)
                if suffix == "ion" and not stem.endswith(("s", "t")):
                    return
                if self.m(stem) > 1:
                    self.word = stem
                return

    def step5a(self):
        if self.word.endswith("e"):
            stem = self.word[:-1]
            if self.m(stem) > 1 or (self.m(stem) == 1 and not self.cvc(stem)):
                self.word = stem

    def step5b(self):
        if self.m() > 1 and self.double_consonant() and self.word.endswith("l"):
            self.word = self.word[:-1]

    def stem(self):
        # words of one or two letters are left as they are
        if len(self.word) <= 2:
            return self.word
        self.step1a()
        self.step1b()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5a()
        self.step5b()
        return self.word


def porter_stemmer(sentences):
    # remove special characters, "Bones's" -> "bones s"
    words = re.sub("[^A-Za-z0-9]+", " ", sentences).split()
    return [PorterStemmer(word).stem() for word in words]


def main():
    text = "This was not the map we found in Billy Bones’s chest, but an accurate copy, complete in all things names and heights and soundings with the single exception of the red crosses and the written notes."
    # text = "Multidimensional characterization"
    # text = "found"
    # text = "thing"
    print(" ".join(porter_stemmer(text)))


if __name__ == "__main__":
    main()
