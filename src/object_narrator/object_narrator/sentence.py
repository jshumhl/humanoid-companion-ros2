"""Turn detected class names into a spoken Chinese sentence."""

from collections import Counter

from .vocab import lookup

_DIGITS = "零一二三四五六七八九"

_TEMPLATES = {
    #         prefix,     conjunction, "two",  nothing seen
    "zh-CN": ("我看到了", "和", "两", "我没有看到任何东西。"),
    "zh-TW": ("我看到了", "和", "兩", "我沒有看到任何東西。"),
    "zh-HK": ("我見到", "同", "兩", "我乜嘢都見唔到。"),
}


def count_classes(class_names):
    """Deduplicate class names into [(name, count)], most frequent first.

    Ties keep the order in which classes were first seen.
    """
    counts = Counter(class_names)
    return sorted(counts.items(), key=lambda item: -item[1])


def chinese_number(n, two):
    """Chinese numeral for a count before a measure word (1-99; digits beyond)."""
    if n == 2:
        return two
    if n < 10:
        return _DIGITS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return ("" if tens == 1 else _DIGITS[tens]) + "十" + (_DIGITS[ones] if ones else "")
    return str(n)


def build_sentence(counts, language):
    """Build a sentence like "我看到了两个人、一只狗和一张椅子。"."""
    prefix, conjunction, two, nothing = _TEMPLATES[language]
    if not counts:
        return nothing

    phrases = []
    for english_name, count in counts:
        name, measure = lookup(english_name, language)
        phrases.append(f"{chinese_number(count, two)}{measure}{name}")

    if len(phrases) == 1:
        listing = phrases[0]
    else:
        listing = "、".join(phrases[:-1]) + conjunction + phrases[-1]
    return f"{prefix}{listing}。"
