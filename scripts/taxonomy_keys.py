"""
Taxonomy front matter keys and keeping their values identical to English.

A taxonomy value is an identifier, not a label: Hugo groups pages by the exact
string, so every language has to carry the English value. A translated value
("Traffic- und Promotion-Kanäle" for "Traffic and Promotion Channels") becomes
a separate term, and the page drops out of its term page and every list built
from it. Visible labels are translated on the term pages themselves
(content/<lang>/<taxonomy>/<term>/_index.md).

The translation flow translates these values with the rest of the front
matter, so translate_with_flowhunt.py restores them from the English source
after each page, and sync_content_attributes.py keeps them in step afterwards.
The keys come from the site's own taxonomy config, so each site protects
exactly the taxonomies it has.
"""

import re
from pathlib import Path

try:
    import tomllib

    def _toml_loads(text):
        return tomllib.loads(text)
except ModuleNotFoundError:  # Python < 3.11
    import toml

    def _toml_loads(text):
        return toml.loads(text)


# Hugo's taxonomies when a site configures none.
HUGO_DEFAULT_TAXONOMIES = ['tags', 'categories']

# Files that can hold the taxonomy config, in Hugo's lookup order. The first
# one that defines taxonomies wins.
_CONFIG_CANDIDATES = [
    ('config/_default/taxonomies.toml', None),
    ('config/_default/hugo.toml', 'taxonomies'),
    ('config/_default/config.toml', 'taxonomies'),
    ('hugo.toml', 'taxonomies'),
    ('config.toml', 'taxonomies'),
]

_FRONTMATTER_RE = re.compile(r'^(﻿?\s*\+\+\+[ \t]*\r?\n)(.*?)(\r?\n\+\+\+)', re.S)

_KEYS_CACHE = {}


def load_taxonomy_keys(hugo_root):
    """
    Front matter keys of the site's taxonomies (the plural names, e.g.
    "faq-categories"), read from its Hugo config. Falls back to Hugo's
    defaults when the site does not configure taxonomies.
    """
    hugo_root = Path(hugo_root)
    if hugo_root in _KEYS_CACHE:
        return _KEYS_CACHE[hugo_root]

    keys = None
    for rel_path, table in _CONFIG_CANDIDATES:
        path = hugo_root / rel_path
        if not path.is_file():
            continue
        try:
            data = _toml_loads(path.read_text(encoding='utf-8-sig'))
        except Exception:
            continue
        if table is not None:
            data = data.get(table)
        if data:
            keys = [str(value) for value in data.values() if value]
            break

    if keys is None:
        keys = list(HUGO_DEFAULT_TAXONOMIES)
    _KEYS_CACHE[hugo_root] = keys
    return keys


def _split(text):
    """(head, frontmatter, tail) around the +++ fences, or None."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3) + text[match.end():]


def _top_level_end(lines):
    """Index of the first table header; top-level keys sit before it."""
    for i, line in enumerate(lines):
        if line.lstrip().startswith('['):
            return i
    return len(lines)


def _find_assignment(lines, key, end):
    """
    (start, stop) line range of the top-level assignment of ``key``, or None.
    The range grows until it parses, so multi-line arrays are covered.
    """
    key_re = re.compile(r'^\s*(?:%s|"%s"|\'%s\')\s*=' % ((re.escape(key),) * 3))
    for start in range(end):
        if not key_re.match(lines[start]):
            continue
        for stop in range(start + 1, end + 1):
            try:
                if key in _toml_loads('\n'.join(lines[start:stop]) + '\n'):
                    return start, stop
            except Exception:
                continue
        return None
    return None


def restore_taxonomies(translated_text, source_text, keys):
    """
    Put the source page's taxonomy values back into a translated page.

    Only the assignment lines of the taxonomy keys change; the rest of the
    translated front matter is kept byte for byte. A key the source does not
    have is left alone. If the result would not parse, the translated text is
    returned unchanged.

    Returns (text, restored_keys).
    """
    source = _split(source_text)
    target = _split(translated_text)
    if not source or not target:
        return translated_text, []

    try:
        source_data = _toml_loads(source[1])
        target_data = _toml_loads(target[1])
    except Exception:
        return translated_text, []

    source_lines = [line.rstrip('\r') for line in source[1].split('\n')]
    lines = target[1].split('\n')
    # Inserted lines take the translated file's line ending.
    eol = '\r' if lines and lines[0].endswith('\r') else ''
    restored = []

    for key in keys:
        if key not in source_data or target_data.get(key) == source_data[key]:
            continue
        source_range = _find_assignment(source_lines, key, _top_level_end(source_lines))
        if source_range is None:
            continue
        replacement = [line + eol for line in source_lines[source_range[0]:source_range[1]]]

        target_range = _find_assignment(lines, key, _top_level_end(lines))
        if target_range is not None:
            lines[target_range[0]:target_range[1]] = replacement
        else:
            # Before the first table header, or the key would land inside it.
            at = _top_level_end(lines)
            while at > 0 and not lines[at - 1].strip():
                at -= 1
            lines[at:at] = replacement
        restored.append(key)

    if not restored:
        return translated_text, []

    frontmatter = '\n'.join(lines)
    try:
        result_data = _toml_loads(frontmatter + '\n')
    except Exception:
        return translated_text, []
    if any(result_data.get(key) != source_data[key] for key in restored):
        return translated_text, []

    return target[0] + frontmatter + target[2], restored
