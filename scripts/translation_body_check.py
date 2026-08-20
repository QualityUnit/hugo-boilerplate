#!/usr/bin/env python3
"""
Structural checks on the body of a translated Hugo page.

The frontmatter guard in translate_with_flowhunt catches broken TOML, but
nothing looked at the markdown below it - and that is where the translation
flow breaks things just as often. Two real cases from run #5, each a single
corrupted character that took down a whole language build:

    fi:661   {{< /table-simple > %}      instead of {{< /table-simple >}}
    tl:~164  a JSON row object closed with ']' instead of '}'

Both are deterministic to detect: shortcode delimiters either balance or they
do not, and a JSON parameter either parses or it does not. Neither is
deterministic to repair - a missing brace could belong in several places - so
this module only reports. The caller writes the file anyway and says loudly
what is wrong, on the same principle as the frontmatter guard: a broken file
can be fixed in the PR, a missing one has to be translated again.
"""

import json
import re

# Markdown ATX headings, ## and deeper. The count per level is a structural
# invariant: a translation renames a heading, it does not add or remove one.
# Length is NOT such an invariant - Russian runs ~65% longer than English on the
# same page and Vietnamese ~45%, so comparing sizes would only produce noise.
_HEADING_RE = re.compile(r'^(#{2,6})\s+\S', re.M)

# Opening shortcode: {{< name ... >}} or {{% name ... %}}, closing: {{< /name >}}
_OPEN_RE = re.compile(r'\{\{[<%]\s*(?!/)([a-zA-Z0-9_-]+)')
_CLOSE_RE = re.compile(r'\{\{[<%]\s*/\s*([a-zA-Z0-9_-]+)')

# A shortcode delimiter that opens but never closes properly, e.g. "> %}".
_MALFORMED_DELIM_RE = re.compile(r'\{\{[<%][^}]*?(?:>\s*%\}|%\s*>\}\}|>\s*\}(?!\}))')

# Shortcodes that take no closing tag. Anything else that opens is expected to
# close; a shortcode used both ways in one project would show up as a false
# positive, which is why the result is a report and not a hard failure.
SELF_CLOSING_HINT = re.compile(r'/\s*[>%]\}\}\s*$')


def _json_blocks(text):
    """
    Yield (start_line, block) for every brace-delimited JSON-looking block.

    Deliberately crude: it walks braces from each `{` that is followed by a
    quoted key on the same or next line, which is what the table-simple style
    parameters look like. Anything that is not JSON simply fails to parse and
    is skipped by the caller.
    """
    for m in re.finditer(r'\{\s*\n?\s*"', text):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    yield text[:start].count('\n') + 1, text[start:i + 1]
                    break
            elif c in ']' and depth == 0:
                break


def _heading_counts(text):
    counts = {}
    for m in _HEADING_RE.finditer(text or ''):
        level = len(m.group(1))
        counts[level] = counts.get(level, 0) + 1
    return counts


def check_body(text, reference=None):
    """
    Report structural problems in a translated body.

    ``reference`` is the English source. When given, shortcode counts are
    compared against it rather than judged in isolation, which removes the
    guesswork about which shortcodes are self-closing: what matters is that the
    translation has the same structure as the page it was translated from.

    Returns a list of human-readable problem strings, empty when clean.
    """
    problems = []

    for m in _MALFORMED_DELIM_RE.finditer(text or ''):
        line = (text[:m.start()].count('\n')) + 1
        problems.append(f'line {line}: malformed shortcode delimiter {m.group(0)[:40]!r}')

    # Missing headings mean the flow stopped early. That is how 24 of 28
    # translations of one page lost their last two sections - the text simply
    # ends mid-list, which is still valid markdown, so the build passed and the
    # loss shipped unnoticed.
    if reference is not None:
        mine = _heading_counts(text)
        theirs = _heading_counts(reference)
        for level in sorted(set(mine) | set(theirs)):
            got, want = mine.get(level, 0), theirs.get(level, 0)
            if got != want:
                problems.append(
                    f'has {got} H{level} heading(s), the english source has {want}'
                    + (' - the translation looks truncated' if got < want else ''))

    def counts(t):
        opens = {}
        closes = {}
        for m in _OPEN_RE.finditer(t or ''):
            opens[m.group(1)] = opens.get(m.group(1), 0) + 1
        for m in _CLOSE_RE.finditer(t or ''):
            closes[m.group(1)] = closes.get(m.group(1), 0) + 1
        return opens, closes

    opens, closes = counts(text)

    if reference is not None:
        ref_opens, ref_closes = counts(reference)
        for name in sorted(set(ref_opens) | set(opens)):
            if opens.get(name, 0) != ref_opens.get(name, 0):
                problems.append(
                    f'shortcode "{name}" opens {opens.get(name, 0)}x, '
                    f'{ref_opens.get(name, 0)}x in the english source')
        for name in sorted(set(ref_closes) | set(closes)):
            if closes.get(name, 0) != ref_closes.get(name, 0):
                problems.append(
                    f'shortcode "{name}" closes {closes.get(name, 0)}x, '
                    f'{ref_closes.get(name, 0)}x in the english source')
    else:
        for name, n in sorted(closes.items()):
            if opens.get(name, 0) != n:
                problems.append(
                    f'shortcode "{name}" opens {opens.get(name, 0)}x but closes {n}x')

    # JSON parameters. Only blocks that parse in the English source are
    # required to parse here, so a page whose source is already odd does not
    # produce noise.
    ref_ok = 0
    if reference is not None:
        for _, block in _json_blocks(reference):
            try:
                json.loads(block)
                ref_ok += 1
            except ValueError:
                pass

    bad_json = []
    ok_json = 0
    for line, block in _json_blocks(text or ''):
        try:
            json.loads(block)
            ok_json += 1
        except ValueError as exc:
            bad_json.append((line, str(exc)[:80]))

    if reference is None or ref_ok:
        for line, err in bad_json:
            problems.append(f'line {line}: JSON parameter does not parse - {err}')

    return problems
