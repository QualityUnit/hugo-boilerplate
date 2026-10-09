#!/usr/bin/env python3
"""Word counts for Japanese and Chinese anchor labels (QualityUnit/web-issues#4254).

In Latin-script languages an anchor needs two words, which the spaces between words
make easy to count. Japanese and Chinese write no spaces, so without a word
segmenter a single word such as ライブ ("live", cut out of ライブチャット "live chat")
or 重要性 ("importance") passes as an anchor. These counters split a label into words
so the generator can apply the same two-word rule:

- Japanese: fugashi (MeCab) with the unidic-lite dictionary. Particles, auxiliary
  verbs, suffixes and dependent verbs are not words: 顧客満足度 = 顧客 / 満足 / 度 =
  two or more words, 生産性 = 生産 (/ 性) and 統合する = 統合 (/ する) = one word.
  Prefixes do count: 中小企業 ("small business"), 不動産 ("real estate") and お礼状
  ("thank-you letter") are two words in English too; excluding prefixes would drop
  them and catch only rare one-word cases such as 再購入 ("repurchase").
- Chinese: jieba, then every word of three or more characters split again into the
  fewest dictionary words, because jieba's dictionary keeps set phrases such as
  客户服务 ("customer service") or 知识库 ("knowledge base") whole. A three-character
  word ending in a suffix (性, 化, 者, 器, 度, 员) is one word: 服务器 ("server"),
  重要性 ("importance").

Measured on the 362 jp and 340 zh-hans anchors of a full generator run (2026-10):
the rule drops one-word anchors only, apart from a few dictionary compounds the
site lists as exceptions (``cjk_multiword_terms`` in generator.yaml).

A library is loaded on the first label it has to count (HTML path, CJK language
only), so other languages and the markdown path never need it. A missing one raises
RuntimeError, which ends the whole generator run (languages already processed keep
their output) instead of silently falling back to a different rule.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from linkbuilding_html import CJK_RE

_JA_NOT_WORDS = frozenset({"助詞", "助動詞", "接尾辞", "補助記号", "記号", "空白"})
_ZH_SUFFIXES = frozenset("性化者器度员")
_ZH_FUNCTION_CHARS = frozenset("的了和与及在是之")

SCRIPTS = ("ja", "zh")


class CjkWordCounter:
    """The words of a Japanese (``ja``) or Chinese (``zh``) label.

    ``groups(text)`` returns, in order, each segmenter token as written together with
    the words it counts as: (自動化, (自動,)) — the suffix 化 is not a word but stays
    part of the token; (工作流, (工作, 流)); particles and symbols form no group.
    ``words(text)`` is the flat list of words, ``count(text)`` their number. The
    generator needs the count for the two-word rule and the groups to check that not
    every part of a label is generic or a brand: a generic term may be listed in its
    written form (自動化) or as its words.
    """

    def __init__(self, script: str) -> None:
        if script not in SCRIPTS:
            raise ValueError(f"CJK script must be one of {', '.join(SCRIPTS)}, got {script!r}")
        self.script = script
        self._loaded = False
        self.groups = lru_cache(maxsize=None)(self._groups)

    def words(self, text: str) -> tuple[str, ...]:
        return tuple(word for _, words in self.groups(text) for word in words)

    def count(self, text: str) -> int:
        return len(self.words(text))

    def _load(self) -> None:
        if self.script == "ja":
            try:
                import fugashi
                self._tagger = fugashi.Tagger()
            except Exception as exc:  # ImportError, or fugashi without a dictionary
                raise RuntimeError(
                    "Japanese anchors need fugashi with the unidic-lite dictionary "
                    "(pip install fugashi unidic-lite, see requirements.txt): " + str(exc)
                ) from exc
        else:
            try:
                import jieba
            except ImportError as exc:
                raise RuntimeError("Chinese anchors need jieba (pip install jieba, see requirements.txt)") from exc
            jieba.setLogLevel(logging.WARNING)
            jieba.dt.check_initialized()
            self._jieba = jieba
            self._freq = jieba.dt.FREQ
        self._loaded = True

    def _groups(self, text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if not self._loaded:
            self._load()
        if self.script == "ja":
            # A word starts a group; a suffix or a dependent verb (pos2 非自立可能: する
            # in 統合する "integrate") is no word of its own but joins the written form
            # of the group before it; particles, auxiliaries and symbols end a group.
            groups: list[list] = []
            attach = False
            for token in self._tagger(text):
                pos1, pos2 = token.feature.pos1, token.feature.pos2
                if pos1 not in _JA_NOT_WORDS and pos2 != "非自立可能":
                    groups.append([token.surface, [token.surface]])
                    attach = True
                elif attach and (pos1 == "接尾辞" or pos2 == "非自立可能"):
                    groups[-1][0] += token.surface
                else:
                    attach = False
            return tuple((raw, tuple(words)) for raw, words in groups)
        zh_groups: list[tuple[str, tuple[str, ...]]] = []
        for word in self._jieba.lcut(text):
            if not word.strip():
                continue
            if not CJK_RE.search(word):  # a Latin word or number ("LiveAgent", "SLA")
                pieces = [word]
            elif len(word) == 3 and word[-1] in _ZH_SUFFIXES:
                pieces = [word[:2]]
            elif len(word) > 2:
                pieces = self._dictionary_pieces(word)
            else:
                pieces = [word]
            kept = tuple(p for p in pieces if not (len(p) == 1 and (p in _ZH_SUFFIXES or p in _ZH_FUNCTION_CHARS)))
            if kept:
                zh_groups.append((word, kept))
        return tuple(zh_groups)

    def _dictionary_pieces(self, word: str) -> list[str]:
        """The fewest dictionary words shorter than ``word`` that make it up (single characters always fit)."""
        best: list[list[str] | None] = [None] * (len(word) + 1)
        best[0] = []
        for i in range(len(word)):
            if best[i] is None:
                continue
            for j in range(i + 1, len(word) + 1):
                piece = word[i:j]
                if j - i > 1 and (piece == word or not self._freq.get(piece)):
                    continue
                candidate = best[i] + [piece]
                if best[j] is None or len(candidate) < len(best[j]):
                    best[j] = candidate
        return best[len(word)] or [word]
