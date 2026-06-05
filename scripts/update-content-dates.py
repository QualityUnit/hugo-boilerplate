#!/usr/bin/env python3
"""
update-content-dates.py

Pre-commit hook: updates the `date` frontmatter attribute when the main body
(post-frontmatter content) of an English markdown file changes.

Only content/en/ files are touched — sync_content_attributes.py propagates
the updated date to all other language files.

Usage (called automatically by .husky/pre-commit):
  python3 themes/boilerplate/scripts/update-content-dates.py
"""

import re
import subprocess
import datetime
from pathlib import Path

FRONTMATTER_RE = re.compile(r'^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n?(.*)', re.DOTALL)
DATE_LINE_RE = re.compile(r'^date\s*=\s*.*$', re.MULTILINE)

CONTENT_DIRS = ('content/en/',)


def extract_body(text: str) -> str:
    m = FRONTMATTER_RE.match(text)
    return m.group(2) if m else text


def git(*args) -> tuple[int, str]:
    r = subprocess.run(['git', *args], capture_output=True, text=True)
    return r.returncode, r.stdout


def get_staged_files() -> list[str]:
    _, out = git('diff', '--cached', '--name-only', '--diff-filter=ACM')
    return [
        f for f in out.strip().splitlines()
        if f.endswith('.md') and any(f.startswith(d) for d in CONTENT_DIRS)
    ]


def read_blob(ref: str) -> str | None:
    code, out = git('show', ref)
    return out if code == 0 else None


def update_date(content: str, new_date: str) -> str:
    m = re.match(r'^(\+\+\+\s*\n)(.*?)(\n\+\+\+\s*\n?)(.*)', content, re.DOTALL)
    if not m:
        return content

    opening, raw_fm, closing, body = m.groups()
    new_line = f'date = "{new_date}"'

    if DATE_LINE_RE.search(raw_fm):
        new_fm = DATE_LINE_RE.sub(new_line, raw_fm)
    else:
        lines = raw_fm.split('\n')
        insert_at = next(
            (i + 1 for i, l in enumerate(lines) if re.match(r'^title\s*=', l.strip())),
            1
        )
        lines.insert(insert_at, new_line)
        new_fm = '\n'.join(lines)

    return f'{opening}{new_fm}{closing}{body}'


def main() -> None:
    _, repo_root = git('rev-parse', '--show-toplevel')
    repo_root = repo_root.strip()

    staged_files = get_staged_files()
    if not staged_files:
        return

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    updated: list[str] = []

    for rel_path in staged_files:
        staged_text = read_blob(f':{rel_path}')
        if staged_text is None:
            continue

        staged_body = extract_body(staged_text)

        head_text = read_blob(f'HEAD:{rel_path}')
        if head_text is not None and extract_body(head_text) == staged_body:
            continue  # only frontmatter changed — skip

        file_path = Path(repo_root) / rel_path
        try:
            disk_content = file_path.read_text(encoding='utf-8')
        except OSError:
            continue

        new_content = update_date(disk_content, now)
        if new_content == disk_content:
            continue

        file_path.write_text(new_content, encoding='utf-8')
        subprocess.run(['git', 'add', str(file_path)], check=True)
        updated.append(rel_path)

    if updated:
        print(f'[update-content-dates] Updated date in {len(updated)} file(s):')
        for f in updated:
            print(f'  {f}')


if __name__ == '__main__':
    main()
