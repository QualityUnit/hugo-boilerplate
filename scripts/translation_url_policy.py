#!/usr/bin/env python3
"""
URL/slug policy for translated Hugo pages.

The translation flow returns a whole translated markdown file. Left alone it
copies the English ``url`` verbatim into every language, so a German page ends
up at ``/features/ai-sales-assistant/`` while every other German feature page
lives under ``/funktionen/``. It also never emits ``aliases``.

This module owns the decision of what ``url`` a translated page gets, and it
is deliberately deterministic: the flow supplies the *words*, this file
supplies the *rules*. Anything the flow returns that cannot be normalised into
a clean ASCII slug falls back to the English slug rather than shipping a
broken URL — a slug is effectively permanent once published, so the safe
failure mode is "same as English", never "mangled".

Policy (agreed 17 Aug 2026):

* Localize the slug in every section EXCEPT those whose slug is a proper name
  or a campaign code (integration names, author names, landing-page codes).
* Non-latin scripts use latin transliteration, which is what the site already
  does (``single-sign-on`` -> ``tasgeel-eldekhol-elohady`` in Arabic).
  Japanese is the exception: no Japanese slug has ever been translated.
* A page that already had a translated URL before it was re-translated keeps
  that URL, restored from ``data/translation_urls/<section>.json``. Losing a
  published URL is the one outcome worth preventing at any cost.
* The section prefix always comes from ``<lang>/<section>/_index.md``, never
  from the flow.
"""

import json
import re
import unicodedata
from pathlib import Path

try:
    import tomllib

    def _toml_load(fh):
        return tomllib.load(fh)
except ImportError:  # Python < 3.11
    import toml

    def _toml_load(fh):
        return toml.loads(fh.read().decode('utf-8'))

# ---------------------------------------------------------------------------
# Per-project configuration
#
# This module ships in a submodule shared by every QualityUnit Hugo site, and
# the sites do not agree: LiveAgent and PostAffiliatePro give each language its
# own domain, FlowHunt and amicited put every language behind a /<lang>/ prefix
# on one domain, and each has its own section names. "Japanese keeps the
# English slug" is a LiveAgent decision, not a fact about Hugo.
#
# So nothing is hardcoded here and nothing happens by default. A project opts
# in by committing data/translation-url-policy.toml; without that file this
# module does nothing at all and translation behaves exactly as it did before.
# That way a repo we have never seen cannot be reshaped by a submodule bump.
# ---------------------------------------------------------------------------

POLICY_FILE = 'data/translation-url-policy.toml'

DEFAULTS = {
    'enabled': False,
    'english_slug_languages': [],
    'passthrough_sections': [],
    'brand_sections': [],
    'brand_suffixes': [],
    'brand_extra': [],
    'max_slug_length': 80,
    'language_prefix': None,   # None = auto-detect from the declared baseURLs
    'aliases': True,
}


# Transliteration tables. These are facts about writing systems, not project
# policy, so they stay in the theme.

# Expansions applied before ASCII folding, so German umlauts survive as the
# digraphs readers expect (NFKD alone would give "uber", not "ueber").
LANG_CHAR_MAP = {
    'de': {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'},
    'sv': {'å': 'aa', 'ä': 'ae', 'ö': 'oe'},
    'da': {'æ': 'ae', 'ø': 'oe', 'å': 'aa'},
    'no': {'æ': 'ae', 'ø': 'oe', 'å': 'aa'},
    'fi': {'ä': 'ae', 'ö': 'oe'},
}

# Letters NFKD does NOT decompose, because they are distinct letters rather
# than a base plus a mark. Without these a Polish slug containing "ł" is
# rejected wholesale and the page silently keeps its English slug -
# "zgloszen" folds to nothing, not to "zgloszen".
UNDECOMPOSABLE = {
    'ł': 'l', 'Ł': 'l',
    'ø': 'o', 'Ø': 'o',
    'æ': 'ae', 'Æ': 'ae',
    'œ': 'oe', 'Œ': 'oe',
    'đ': 'd', 'Đ': 'd', 'ð': 'd', 'Ð': 'd',
    'ß': 'ss',
    'þ': 'th', 'Þ': 'th',
    'ħ': 'h', 'ı': 'i', 'ŋ': 'n', 'ŧ': 't', 'ſ': 's', 'ə': 'e',
}

_BASEURL_RE = re.compile(r'^\s*baseURL\s*=\s*[\'"]([^\'"]+)[\'"]', re.M)

_URL_RE = re.compile(r'^url\s*=\s*"([^"]*)"', re.M)
_ALIASES_RE = re.compile(r'^aliases\s*=\s*\[[^\]]*\]', re.M)
_FRONTMATTER_RE = re.compile(r'^\+\+\+\s*\n(.*?)\n\+\+\+', re.S)


# ---------------------------------------------------------------------------
# Small parsing helpers. These read with regex on purpose: the frontmatter of a
# freshly translated file is not guaranteed to parse as TOML yet (that is what
# the repair pass upstream is for), and the URL line is well-formed even when
# some FAQ answer three screens down is not.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Site shape
#
# Two deployment models share this script, and they need different URLs:
#
#   own domain per language (LiveAgent, PostAffiliatePro)
#       liveagent.de/funktionen/regeln/ - no language prefix, and each language
#       is built on its own (config_<lang> + HUGO_DEFAULTCONTENTLANGUAGE), so
#       an English-path alias on a German page is safe.
#
#   one domain, language subfolders (FlowHunt, amicited, photomaticai)
#       flowhunt.io/de/ai-flow-templates/ - the prefix is part of every url,
#       and all languages are built into one public/ tree.
#
# The distinction is not cosmetic. In the subfolder model an alias without the
# language prefix silently overwrites the English page: a German page carrying
#     aliases = ["/features/rules/"]
# publishes a redirect at /features/rules/index.html, replacing the English
# page that lives there. Hugo emits no warning for this - verified on Hugo
# 0.160.1 with a two-language test site. FlowHunt content already contains such
# aliases today.
# ---------------------------------------------------------------------------

_PREFIX_CACHE = {}


_POLICY_CACHE = {}


def load_policy(hugo_root):
    """
    Read data/translation-url-policy.toml for this project.

    Returns None when the file is absent, unreadable, or does not set
    ``enabled = true``. A None policy means this module stands down completely -
    the caller writes the translated file exactly as the flow returned it.
    """
    key = str(hugo_root)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    policy = None
    path = Path(hugo_root) / POLICY_FILE
    if path.exists():
        try:
            with open(path, 'rb') as fh:
                raw = _toml_load(fh)
        except Exception as exc:
            print(f'[URL] cannot read {POLICY_FILE}: {exc} - url policy disabled')
            raw = None
        if raw is not None:
            if raw.get('enabled') is True:
                policy = dict(DEFAULTS)
                policy.update({k: v for k, v in raw.items() if k in DEFAULTS})
                policy['passthrough_sections'] = set(policy['passthrough_sections'] or [])
                policy['english_slug_languages'] = set(policy['english_slug_languages'] or [])
            else:
                print(f'[URL] {POLICY_FILE} present but enabled is not true - url policy disabled')

    _POLICY_CACHE[key] = policy
    return policy


def uses_language_prefix(hugo_root):
    """
    True when URLs on this site carry a /<lang>/ prefix.

    Derived the same way sync_translation_urls does it: if the languages
    declare two or more distinct hostnames, each language owns a domain and
    needs no prefix. Anything else (one baseURL, or none declared) means the
    languages share a domain and are separated by a subfolder.
    """
    key = str(hugo_root)
    if key in _PREFIX_CACHE:
        return _PREFIX_CACHE[key]

    hosts = set()
    for rel in ('config/_default/languages.toml', 'config/_default/hugo.toml',
                'config/_default/config.toml', 'hugo.toml', 'config.toml'):
        path = Path(hugo_root) / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            continue
        for raw in _BASEURL_RE.findall(text):
            host = raw.split('//', 1)[-1].split('/', 1)[0].strip()
            if host:
                hosts.add(host.lower())

    if len(hosts) >= 2:
        result = False          # a domain per language, no prefix needed
    elif len(hosts) == 1:
        result = True           # one declared domain, languages in subfolders
    else:
        result = None           # nothing declared - refuse to guess
    _PREFIX_CACHE[key] = result
    return result


def read_url(text):
    """Return the ``url`` value from a markdown file's frontmatter, or None."""
    fm = _FRONTMATTER_RE.match(text or '')
    if not fm:
        return None
    m = _URL_RE.search(fm.group(1))
    return m.group(1) if m else None


def ensure_slashes(url):
    """Normalise an internal URL to /leading/ and /trailing/ slashes."""
    if not url:
        return url
    if not url.startswith('/'):
        url = '/' + url
    if not url.endswith('/'):
        url = url + '/'
    return url


def split_url(url):
    """Split ``/a/b/c/`` into (``a/b``, ``c``)."""
    parts = [p for p in (url or '').strip('/').split('/') if p]
    if not parts:
        return '', ''
    return '/'.join(parts[:-1]), parts[-1]


def normalize_slug(raw, lang, max_length=80):
    """
    Turn whatever the flow returned into a safe ASCII slug.

    Returns '' when nothing usable survives — the caller then falls back to the
    English slug. Rejecting is the point: a slug that still contains non-ASCII
    after folding means the flow returned native script, and shipping a
    percent-encoded URL by accident is worse than shipping the English one.
    """
    if not raw:
        return ''

    text = str(raw).strip().lower()

    for src, dst in LANG_CHAR_MAP.get(lang, {}).items():
        text = text.replace(src, dst)
    for src, dst in UNDECOMPOSABLE.items():
        if src in text:
            text = text.replace(src, dst)

    # Strip combining marks: é -> e, ā -> a. Characters with no ASCII
    # decomposition (Greek, Cyrillic, CJK, Arabic) simply survive and are
    # caught by the ASCII check below.
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    if any(ord(c) > 127 for c in text):
        return ''

    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    text = re.sub(r'-{2,}', '-', text)

    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit('-', 1)[0].strip('-')

    return text


# ---------------------------------------------------------------------------
# Site lookups
# ---------------------------------------------------------------------------

_BRAND_CACHE = {}

# Words that appear inside an integration or competitor slug but are ordinary
# vocabulary, so they must not be treated as brands to preserve.
_BRAND_STOPWORDS = {
    'alternative', 'alternatives', 'migration', 'integration', 'integrations',
    'software', 'app', 'apps', 'plugin', 'free', 'vs', 'and', 'for', 'the',
    'chat', 'live', 'help', 'desk', 'email', 'mail', 'phone', 'call', 'sms',
    'crm', 'api', 'cloud', 'web', 'online', 'best', 'top', 'new', 'to', 'of',
    'business', 'virtual', 'assistant', 'agent', 'agents', 'center', 'centre',
    'service', 'services', 'support', 'customer', 'ticket', 'ticketing',
    'system', 'systems', 'tool', 'tools', 'marketing', 'social', 'media',
    'video', 'voice', 'voip', 'sales', 'team', 'inbox', 'form', 'forms',
}


def brand_tokens(content_dir, sections=(), suffixes=(), extra=()):
    """
    Product and company names that appear in this site's own URLs.

    Built from the site's own content rather than a hand-kept vendor list. The
    project says where to look: ``brand_sections`` are sections whose page names
    are product names (an integrations directory), and ``brand_suffixes`` are
    the endings of comparison pages named after a competitor
    (``<vendor>-alternative``). That makes the vocabulary self-maintaining - a
    new integration page teaches it a new brand - without assuming any
    particular site layout.
    """
    key = (str(content_dir), tuple(sections or ()), tuple(suffixes or ()), tuple(extra or ()))
    if key in _BRAND_CACHE:
        return _BRAND_CACHE[key]

    tokens = {t.lower() for t in (extra or ()) if t}
    en_dir = Path(content_dir) / 'en'

    for section in sections or ():
        for path in (en_dir / section).glob('*.md'):
            if path.name == '_index.md':
                continue
            for token in path.stem.lower().split('-'):
                if len(token) > 2 and token not in _BRAND_STOPWORDS:
                    tokens.add(token)

    for path in en_dir.glob('*.md'):
        stem = path.stem.lower()
        for suffix in suffixes or ():
            if not stem.endswith(suffix):
                continue
            head = stem[: -len(suffix)]
            # "zendesk-talk-alternative" names the product "zendesk talk", so
            # index both words rather than the joined string, which would never
            # match a hyphen-split slug.
            for token in head.split('-'):
                if len(token) > 2 and token not in _BRAND_STOPWORDS:
                    tokens.add(token)
            break

    _BRAND_CACHE[key] = tokens
    return tokens


def section_prefix(content_dir, lang, section, fallback, lang_prefix=False):
    """
    The URL path a section lives under in one language, taken from that
    language's own ``<section>/_index.md``.

    On subfolder sites that index already carries the prefix
    (``/de/ai-flow-templates/``), so nothing extra is needed. It is only the
    fallback - a language that has no section index yet - that must add the
    prefix itself, otherwise the page would be published on the English path
    and collide with the English page.
    """
    if not section:
        fallback = (fallback or '').strip('/')
        if lang_prefix and fallback.split('/')[0] != lang:
            fallback = '/'.join(p for p in (lang, fallback) if p)
        return fallback
    index = Path(content_dir) / lang / section / '_index.md'
    if index.exists():
        try:
            url = read_url(index.read_text(encoding='utf-8-sig'))
        except OSError:
            url = None
        if url:
            return url.strip('/')
    fallback = (fallback or '').strip('/')
    if lang_prefix and fallback.split('/')[0] != lang:
        fallback = '/'.join(p for p in (lang, fallback) if p)
    return fallback


def data_file_for_section(section):
    """
    The translation_urls file a section's records live in.

    Two conventions have to be bridged: content directories are hyphenated
    (``success-stories``, ``customer-support-glossary``) while the generated
    JSON files are underscored, and pages that sit at the top of a language
    (``aircall-alternative.md`` and 207 others) are collected in ``_root``.
    Getting this wrong is silent - the lookup just misses and the page loses
    its published URL.
    """
    if not section:
        return '_root'
    return section.replace('-', '_')


def historical_url(hugo_root, section, rel_path, lang):
    """
    The URL this page had before it was deleted and re-translated.

    ``data/translation_urls/<section>.json`` maps a source path such as
    ``blog/best-trouble-ticket-system.md`` to the published URL of every
    language. It is regenerated from content, so it is only a reliable record
    of the *previous* state — which is exactly what is needed here.
    """
    data_file = (Path(hugo_root) / 'data' / 'translation_urls' /
                 f'{data_file_for_section(section)}.json')
    if not data_file.exists():
        return None
    try:
        with open(data_file, 'rb') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    entry = data.get(str(rel_path).replace('\\', '/'))
    if isinstance(entry, dict):
        url = entry.get(lang)
        if url:
            return ensure_slashes(url)
    return None


def build_url_index(content_dir, lang):
    """
    Every URL already claimed in one language, as {url: source path}.

    Covers both ``url`` and ``aliases`` so a new page cannot silently take an
    address another page already redirects from.
    """
    index = {}
    lang_dir = Path(content_dir) / lang
    if not lang_dir.exists():
        return index
    for path in lang_dir.rglob('*.md'):
        try:
            head = path.read_text(encoding='utf-8-sig')[:8000]
        except OSError:
            continue
        fm = _FRONTMATTER_RE.match(head)
        body = fm.group(1) if fm else head
        m = _URL_RE.search(body)
        if m:
            index.setdefault(ensure_slashes(m.group(1)), str(path))
        a = _ALIASES_RE.search(body)
        if a:
            for alias in re.findall(r'"([^"]+)"', a.group(0)):
                index.setdefault(ensure_slashes(alias), str(path))
    return index


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def resolve_url(en_url, translated_url, rel_path, lang, content_dir, hugo_root,
                url_index=None):
    """
    Decide the final URL for a translated page.

    Returns (url, alias, reason) where ``alias`` is the English URL to redirect
    from (or None) and ``reason`` explains the choice for the run log.
    """
    policy = load_policy(hugo_root)
    if policy is None:
        return None, None, f'no {POLICY_FILE} in this project, url left untouched'

    rel_path = str(rel_path).replace('\\', '/')
    section = rel_path.split('/')[0] if '/' in rel_path else ''
    en_url = ensure_slashes(en_url) if en_url else None

    if not en_url:
        return None, None, 'english page has no explicit url'

    # A section index defines the prefix every page in that section inherits.
    # Renaming it silently moves the whole section, so it stays a human
    # decision - the pipeline only ever fills in leaf pages.
    if rel_path.rsplit('/', 1)[-1] == '_index.md':
        return None, None, 'section index, url left for a human to decide'

    lang_prefix = policy['language_prefix']
    if lang_prefix is None:
        lang_prefix = uses_language_prefix(hugo_root)
    if lang_prefix is None:
        return None, None, ('cannot tell whether this site uses a /<lang>/ url prefix - '
                            'no baseURL declared and language_prefix not set in '
                            f'{POLICY_FILE}; url left untouched')
    en_prefix, en_slug = split_url(en_url)
    prefix = section_prefix(content_dir, lang, section, en_prefix, lang_prefix)

    def assemble(slug):
        return ensure_slashes('/'.join(p for p in (prefix, slug) if p))

    # The English path as it would look inside THIS language. On a per-domain
    # site that is the English URL itself, which is the legacy address the page
    # used to answer on. On a subfolder site it must carry the language prefix,
    # or the alias would land on the English page's own address and replace it.
    english_prefix = en_prefix
    if lang_prefix and en_prefix.split('/')[0] != lang:
        english_prefix = '/'.join(p for p in (lang, en_prefix) if p)
    english_path = ensure_slashes('/'.join(p for p in (english_prefix, en_slug) if p))

    def settle(url, reason):
        """
        Last gate before a URL is handed back: never take an address that
        another page in this language already owns, and never advertise an
        alias that is somebody else's page. Every branch goes through here -
        an early return that skips it is how /blog/gmail-alternative/ ended up
        colliding with a root page during testing.
        """
        if url_index is not None:
            owner = url_index.get(url)
            if owner and not owner.endswith(rel_path):
                fallback = assemble(en_slug)
                if fallback != url and not url_index.get(fallback):
                    url, reason = fallback, f'{reason}; collided with {owner}, used english slug'
                else:
                    return None, None, f'url {url} collides with {owner} and the english slug is taken too'
        alias = english_path if (policy['aliases'] and url != english_path) else None
        if alias and url_index is not None:
            owner = url_index.get(alias)
            if owner and not owner.endswith(rel_path):
                alias = None
        return url, alias, reason

    # 1. A URL this page already had wins over anything the flow invented -
    # unless another page has moved onto that address in the meantime. The
    # map is a snapshot of a past state, and pages get merged and pruned
    # between snapshots, so a stale record must not create a duplicate url
    # (verified: 2 such records in the German blog map alone).
    previous = historical_url(hugo_root, section, rel_path, lang)
    if previous:
        owner = url_index.get(previous) if url_index is not None else None
        if owner and not owner.endswith(rel_path):
            print(f'[URL] {lang}/{rel_path}: previous url {previous} now belongs to '
                  f'{owner}, not restoring')
        else:
            return settle(previous, 'restored previously published url')

    # 2. Sections whose slug is a proper name, and languages that keep English.
    if section in policy['passthrough_sections']:
        return settle(assemble(en_slug), f'section "{section}" keeps the english slug')
    if lang in policy['english_slug_languages']:
        return settle(assemble(en_slug), f'language "{lang}" keeps the english slug')

    # 3. Otherwise use what the flow translated, normalised hard.
    _, raw_slug = split_url(translated_url or '')
    slug = normalize_slug(raw_slug, lang, policy['max_slug_length'])
    reason = 'translated slug'
    if not slug:
        slug = en_slug
        reason = 'flow returned no usable slug, fell back to english'
    elif slug == en_slug:
        reason = 'flow returned the english slug unchanged'

    # A brand name must survive translation. "aircall-alternative" becoming
    # "alternative-zum-anruf" would be a permanently wrong URL, and the flow
    # has no way of knowing which words are products.
    if slug != en_slug:
        expected = set(en_slug.split('-')) & brand_tokens(
            content_dir, policy['brand_sections'], policy['brand_suffixes'],
            policy['brand_extra'])
        lost = expected - set(slug.split('-'))
        if lost:
            slug = en_slug
            reason = f'translated slug dropped brand name(s) {sorted(lost)}, fell back to english'

    # 4. Same gate as every other branch.
    return settle(assemble(slug), reason)


def apply_url(text, url, alias):
    """
    Rewrite the ``url`` line and add ``aliases`` in a translated file, touching
    nothing else. Returns the new text, or the original when there is no
    frontmatter to work with.
    """
    fm = _FRONTMATTER_RE.match(text or '')
    if not fm or not url:
        return text

    block = fm.group(1)
    new_block = block

    if _URL_RE.search(new_block):
        new_block = _URL_RE.sub(lambda _m: f'url = "{url}"', new_block, count=1)
    else:
        lines = new_block.split('\n')
        at = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('title '):
                at = i + 1
                break
        lines.insert(at, f'url = "{url}"')
        new_block = '\n'.join(lines)

    if alias:
        existing = _ALIASES_RE.search(new_block)
        if existing:
            current = re.findall(r'"([^"]+)"', existing.group(0))
            if alias not in current:
                merged = ', '.join(f'"{a}"' for a in current + [alias])
                new_block = _ALIASES_RE.sub(f'aliases = [{merged}]', new_block, count=1)
        else:
            new_block = _URL_RE.sub(
                lambda m: f'{m.group(0)}\naliases = ["{alias}"]', new_block, count=1)

    return text[:fm.start(1)] + new_block + text[fm.end(1):]
