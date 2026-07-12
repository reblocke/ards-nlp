from __future__ import annotations

from typing import Protocol


class StemmerProtocol(Protocol):
    def stem(self, word: str) -> str: ...


class PorterCompatibilityStemmer:
    """Small classic Porter stemmer used to mirror CLAMP dictionary matching.

    Martin Porter's original algorithm and reference implementation are in the public domain.
    This implementation intentionally applies only to ASCII alphabetic tokens; punctuation,
    identifiers, and non-ASCII tokens are case-folded but otherwise preserved.
    """

    _STEP2 = {
        "ational": "ate",
        "tional": "tion",
        "enci": "ence",
        "anci": "ance",
        "izer": "ize",
        "bli": "ble",
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
        "logi": "log",
    }
    _STEP3 = {
        "icate": "ic",
        "ative": "",
        "alize": "al",
        "iciti": "ic",
        "ical": "ic",
        "ful": "",
        "ness": "",
    }
    _STEP4 = (
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
    )

    def stem(self, word: str) -> str:
        value = word.casefold()
        if len(value) < 3 or not value.isascii() or not value.isalpha():
            return value

        value = self._step1a(value)
        value = self._step1b(value)
        value = self._step1c(value)
        value = self._replace_suffix(value, self._STEP2)
        value = self._replace_suffix(value, self._STEP3)
        value = self._step4(value)
        value = self._step5(value)
        return value

    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            return stem + "ee" if self._measure(stem) > 0 else word

        stem: str | None = None
        if word.endswith("ed") and self._contains_vowel(word[:-2]):
            stem = word[:-2]
        elif word.endswith("ing") and self._contains_vowel(word[:-3]):
            stem = word[:-3]
        if stem is None:
            return word

        if stem.endswith(("at", "bl", "iz")):
            return stem + "e"
        if self._ends_double_consonant(stem) and not stem.endswith(("l", "s", "z")):
            return stem[:-1]
        if self._measure(stem) == 1 and self._ends_cvc(stem):
            return stem + "e"
        return stem

    def _step1c(self, word: str) -> str:
        if word.endswith("y") and self._contains_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def _replace_suffix(self, word: str, replacements: dict[str, str]) -> str:
        for suffix in sorted(replacements, key=len, reverse=True):
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacements[suffix]
                return word
        return word

    def _step4(self, word: str) -> str:
        for suffix in sorted(self._STEP4, key=len, reverse=True):
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if suffix == "ion" and (not stem or stem[-1] not in "st"):
                return word
            return stem if self._measure(stem) > 1 else word
        return word

    def _step5(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            measure = self._measure(stem)
            if measure > 1 or (measure == 1 and not self._ends_cvc(stem)):
                word = stem
        if word.endswith("ll") and self._measure(word) > 1:
            word = word[:-1]
        return word

    @staticmethod
    def _is_consonant(word: str, index: int) -> bool:
        char = word[index]
        if char in "aeiou":
            return False
        if char == "y":
            return index == 0 or not PorterCompatibilityStemmer._is_consonant(word, index - 1)
        return True

    @classmethod
    def _measure(cls, word: str) -> int:
        transitions = 0
        previous_vowel = False
        for index in range(len(word)):
            is_vowel = not cls._is_consonant(word, index)
            if previous_vowel and not is_vowel:
                transitions += 1
            previous_vowel = is_vowel
        return transitions

    @classmethod
    def _contains_vowel(cls, word: str) -> bool:
        return any(not cls._is_consonant(word, index) for index in range(len(word)))

    @classmethod
    def _ends_double_consonant(cls, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and cls._is_consonant(word, len(word) - 1)

    @classmethod
    def _ends_cvc(cls, word: str) -> bool:
        if len(word) < 3:
            return False
        last = len(word) - 1
        return (
            cls._is_consonant(word, last)
            and not cls._is_consonant(word, last - 1)
            and cls._is_consonant(word, last - 2)
            and word[last] not in "wxy"
        )
