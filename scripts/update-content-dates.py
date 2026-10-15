#!/usr/bin/env python3
"""
update-content-dates.py

Pre-commit hook: keeps the two dates in the front matter of English pages honest.

  date     the publish date (datePublished). Set while a page is new - not yet on
           the default branch - and never changed after that.
  lastmod  the modified date (dateModified, sitemap <lastmod>). Set when the text
           of a page that is already on the default branch changes.

What counts as a text change: the words of the body (everything after the front
matter). Whitespace, blank lines and Markdown markup (`**`, `#`, `---`, list and
table syntax) are not words, so fixing them changes no date. A change to a word,
a number, a link target or a shortcode does.

A technical change applied to many pages at once (removing a shortcode or a CTA
block site-wide, a new shortcode parameter) is not an edit of their text. Only the
author knows which of the two a commit is, so the author says it:

  SKIP_CONTENT_DATES=1 git commit ...    a technical change: no dates change
  FORCE_CONTENT_DATES=1 git commit ...   a real edit of many pages: lastmod as usual

When one commit changes the text of BULK_THRESHOLD or more published pages and
neither is set, the hook stops the commit and asks.

Times are written in UTC with a `Z`: Hugo reads a time without a zone as UTC, so a
local time would put the page hours into the future and drop it from the build.

Only content/en/ files are touched - sync_content_attributes.py copies `date` to
the translations; `lastmod` stays per language.

Usage (called automatically by .husky/pre-commit):
  python3 themes/boilerplate/scripts/update-content-dates.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import datetime
from pathlib import Path

FRONTMATTER_RE = re.compile(r'^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n?(.*)', re.DOTALL)
# The value up to the end of the line, without a CR, so CRLF files keep theirs.
DATE_LINE_RE = re.compile(r'^date[ \t]*=[^\r\n]*', re.MULTILINE)
LASTMOD_LINE_RE = re.compile(r'^lastmod[ \t]*=[^\r\n]*', re.MULTILINE)
WORD_RE = re.compile(r'\w+')

CONTENT_DIRS = ('content/en/',)
SKIP_ENV = 'SKIP_CONTENT_DATES'
FORCE_ENV = 'FORCE_CONTENT_DATES'
# From this many published pages with changed text in one commit the hook asks
# whether the change is technical or editorial instead of guessing: both kinds
# exist at this size (a CTA removed from 216 pages, headings rewritten on 70).
BULK_THRESHOLD = 10


def extract_body(text: str) -> str:
    m = FRONTMATTER_RE.match(text)
    return m.group(2) if m else text


def extract_front_matter(text: str) -> str:
    m = FRONTMATTER_RE.match(text)
    return m.group(1) if m else ''


def body_words(text: str) -> list[str]:
    # The words of the body, in order - what a reader reads. Whitespace and
    # Markdown markup fall away, so a fix to either compares equal.
    return WORD_RE.findall(extract_body(text))


def lastmod_line(text: str) -> str | None:
    m = LASTMOD_LINE_RE.search(extract_front_matter(text))
    return m.group(0).strip() if m else None


def git(*args, stdin: str | None = None) -> tuple[int, str]:
    # Bytes in and out, decoded here: text=True would turn CRLF into LF, and a
    # file whose blob goes back into the index must keep its line endings.
    r = subprocess.run(
        ['git', *args], capture_output=True,
        input=stdin.encode('utf-8', 'surrogateescape') if stdin is not None else None,
    )
    return r.returncode, r.stdout.decode('utf-8', 'surrogateescape')


def get_staged_files() -> list[str]:
    _, out = git('diff', '--cached', '--name-only', '--diff-filter=ACM')
    return [
        f for f in out.strip().splitlines()
        if f.endswith('.md') and any(f.startswith(d) for d in CONTENT_DIRS)
    ]


def is_merge_or_rebase_in_progress() -> bool:
    # During a merge/rebase, staged files can differ from HEAD purely because
    # they picked up someone else's already-dated changes — not because the
    # author edited them. Bumping the date in that case is wrong.
    #
    # MERGE_HEAD/rebase-merge/rebase-apply live in the per-worktree git dir,
    # which is NOT <repo_root>/.git when repo_root is a linked worktree (there,
    # .git is a file pointing at .../main/.git/worktrees/<name>). Ask git for
    # the real path instead of assuming <repo_root>/.git is a directory.
    code, out = git('rev-parse', '--absolute-git-dir')
    if code != 0:
        return False
    git_dir = Path(out.strip())
    return any((git_dir / name).exists() for name in (
        'MERGE_HEAD', 'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
    ))


def default_branch_ref() -> str | None:
    # The integration branch the commit will eventually land on. A page that is
    # already there is published; a staged body that already matches it did not
    # originate with this author (it arrived via merge, rebase, cherry-pick or a
    # branch update), so it must not be re-dated.
    code, out = git('symbolic-ref', '-q', '--short', 'refs/remotes/origin/HEAD')
    candidates = [out.strip()] if code == 0 and out.strip() else []
    candidates += ['origin/main', 'main']
    for ref in candidates:
        code, _ = git('rev-parse', '--verify', '-q', f'{ref}^{{commit}}')
        if code == 0:
            return ref
    return None


def read_blob(ref: str) -> str | None:
    code, out = git('show', ref)
    # Strip a leading BOM so the `^\+\+\+` front matter regexes below match.
    return out.lstrip('﻿') if code == 0 else None


def set_front_matter_line(content: str, key: str, line_re: re.Pattern, value: str) -> str:
    m = re.match(r'^(\+\+\+\s*\n)(.*?)(\n\+\+\+\s*\n?)(.*)', content, re.DOTALL)
    if not m:
        return content

    opening, raw_fm, closing, body = m.groups()
    new_line = f'{key} = "{value}"'

    if line_re.search(raw_fm):
        new_fm = line_re.sub(new_line, raw_fm, count=1)
    else:
        # lastmod goes right after date, date right after title.
        anchor = r'^date\s*=' if key == 'lastmod' else r'^title\s*='
        lines = raw_fm.split('\n')
        insert_at = next(
            (i + 1 for i, l in enumerate(lines) if re.match(anchor, l.strip())),
            1
        )
        lines.insert(insert_at, new_line + ('\r' if '\r\n' in raw_fm else ''))
        new_fm = '\n'.join(lines)

    return f'{opening}{new_fm}{closing}{body}'


def update_date(content: str, new_date: str) -> str:
    return set_front_matter_line(content, 'date', DATE_LINE_RE, new_date)


def update_lastmod(content: str, new_date: str) -> str:
    return set_front_matter_line(content, 'lastmod', LASTMOD_LINE_RE, new_date)


def stage_content(rel_path: str, content: str) -> None:
    # Write the new version into the index only, so unstaged changes elsewhere
    # in the file stay unstaged (a plain `git add` would commit them too).
    code, out = git('ls-files', '-s', '--', rel_path)
    mode = out.split()[0] if code == 0 and out.strip() else '100644'
    _, sha = git('hash-object', '-w', '--stdin', '--path', rel_path, stdin=content)
    subprocess.run(
        ['git', 'update-index', '--cacheinfo', f'{mode},{sha.strip()},{rel_path}'],
        check=True,
    )


def env_flag(name: str) -> bool:
    return os.environ.get(name, '') not in ('', '0')


def main() -> None:
    _, repo_root = git('rev-parse', '--show-toplevel')
    repo_root = repo_root.strip()

    if env_flag(SKIP_ENV):
        print(f'[update-content-dates] {SKIP_ENV} is set - no dates changed.')
        return

    if is_merge_or_rebase_in_progress():
        return

    staged_files = get_staged_files()
    if not staged_files:
        return

    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    base_ref = default_branch_ref()
    changes: list[tuple[str, str, str]] = []  # (rel_path, key, staged_text)

    for rel_path in staged_files:
        staged_text = read_blob(f':{rel_path}')
        if staged_text is None:
            continue

        staged_words = body_words(staged_text)

        head_text = read_blob(f'HEAD:{rel_path}')
        if head_text is not None and body_words(head_text) == staged_words:
            continue  # no words changed (front matter, whitespace or markup only) — skip

        base_text = read_blob(f'{base_ref}:{rel_path}') if base_ref is not None else None
        if base_text is not None and body_words(base_text) == staged_words:
            continue  # body already on the default branch — not this author's edit

        # Published = already on the default branch. Without one, fall back to HEAD.
        published = base_text is not None if base_ref is not None else head_text is not None

        if published:
            previous = head_text if head_text is not None else base_text
            if lastmod_line(staged_text) != lastmod_line(previous):
                continue  # the author set lastmod by hand in this commit — keep it
            changes.append((rel_path, 'lastmod', staged_text))
        else:
            changes.append((rel_path, 'date', staged_text))

    edited = sum(1 for _, key, _ in changes if key == 'lastmod')
    if edited >= BULK_THRESHOLD and not env_flag(FORCE_ENV):
        # Nothing is written yet: the commit stops as it was staged.
        print(f'[update-content-dates] The text of {edited} published pages changed in one commit.', file=sys.stderr)
        print('  Is it a technical change (a shortcode, a CTA block, a parameter) or an edit of their text?', file=sys.stderr)
        print(f'  Technical - no dates change:   {SKIP_ENV}=1 git commit ...', file=sys.stderr)
        print(f'  Text edit - lastmod on each:   {FORCE_ENV}=1 git commit ...', file=sys.stderr)
        sys.exit(1)

    updated: list[tuple[str, str]] = []
    for rel_path, key, staged_text in changes:
        apply = update_lastmod if key == 'lastmod' else update_date
        new_staged = apply(staged_text, now)
        if new_staged == staged_text:
            continue
        stage_content(rel_path, new_staged)

        # The same front matter line in the working copy, so the file on disk
        # matches what is committed.
        file_path = Path(repo_root) / rel_path
        try:
            # utf-8-sig strips a leading BOM so the front matter regex matches;
            # newline='' keeps CRLF as it is.
            with open(file_path, encoding='utf-8-sig', newline='') as f:
                disk_content = f.read()
        except OSError:
            disk_content = None
        if disk_content is not None:
            new_disk = apply(disk_content, now)
            if new_disk != disk_content:
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(new_disk)

        updated.append((rel_path, key))

    if updated:
        print(f'[update-content-dates] Updated dates in {len(updated)} file(s):')
        for f, key in updated:
            print(f'  {f} ({key})')


if __name__ == '__main__':
    main()
