"""scripts/update-content-dates.py: date is the publish date, lastmod the modified date.

Each test builds a throwaway git repository with an `origin/main`, stages a change
and runs the hook the way .husky/pre-commit does, then reads what it staged.
"""

import datetime
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'update-content-dates.py'

_spec = importlib.util.spec_from_file_location('update_content_dates', SCRIPT)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

PAGE = 'content/en/page.md'
PUBLISHED = '2025-11-24 10:00:00'
UTC_STAMP_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')

BODY = """
## Why switch

LiveAgent answers **every** ticket in one inbox.

{{< blockquote name="David Chandler" >}}
Simply the best.
{{< /blockquote >}}
---

## Pricing

Plans start at $15 per agent.
"""


def page(body=BODY, date=PUBLISHED, lastmod=None, extra=''):
    fm = ['+++', 'title = "Page"']
    if date is not None:
        fm.append(f'date = "{date}"')
    if lastmod is not None:
        fm.append(f'lastmod = "{lastmod}"')
    if extra:
        fm.append(extra)
    fm.append('+++')
    return '\n'.join(fm) + '\n' + body


class Repo:
    """A git repository whose `origin/main` is its first commit."""

    def __init__(self, files):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.git('init', '-q', '-b', 'main')
        self.git('config', 'user.email', 'test@example.com')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'core.autocrlf', 'false')
        for path, text in files.items():
            self.write(path, text)
        self.git('add', '-A')
        self.git('commit', '-q', '-m', 'initial')
        self.git('update-ref', 'refs/remotes/origin/main', 'HEAD')

    def close(self):
        self._tmp.cleanup()

    def git(self, *args):
        r = subprocess.run(['git', *args], cwd=self.root, capture_output=True)
        if r.returncode != 0:
            raise AssertionError(f'git {args} failed: {r.stderr.decode()}')
        return r.stdout.decode('utf-8')

    def write(self, path, text, newline='\n'):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8', newline=newline) as f:
            f.write(text)

    def read(self, path):
        with open(self.root / path, encoding='utf-8', newline='') as f:
            return f.read()

    def staged(self, path):
        return self.git('show', f':{path}')

    def stage(self, path, text, newline='\n'):
        self.write(path, text, newline)
        self.git('add', path)

    def commit(self, message='change'):
        self.git('commit', '-q', '--no-verify', '-m', message)

    def run_hook(self, **env):
        r = self.run_hook_raw(**env)
        if r.returncode != 0:
            raise AssertionError(f'hook failed: {r.stderr}')
        return r.stdout

    def run_hook_raw(self, **env):
        clean = {k: v for k, v in os.environ.items()
                 if k not in ('SKIP_CONTENT_DATES', 'FORCE_CONTENT_DATES')}
        return subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=self.root, capture_output=True, text=True,
            env={**clean, **env},
        )


def front_matter_value(text, key):
    m = re.search(rf'^{key} = "([^"]*)"', text, re.MULTILINE)
    return m.group(1) if m else None


class HookTestCase(unittest.TestCase):
    files = {PAGE: page()}

    def setUp(self):
        self.repo = Repo(dict(self.files))
        self.addCleanup(self.repo.close)

    def assertFreshUtc(self, stamp):
        self.assertRegex(stamp, UTC_STAMP_RE)
        written = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(
            tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        self.assertLess(abs((now - written).total_seconds()), 120)

    def assertUntouched(self, expected):
        self.assertEqual(self.repo.staged(PAGE), expected)
        self.assertEqual(self.repo.read(PAGE), expected)


class PublishedPage(HookTestCase):
    """The page is on origin/main: date never changes, lastmod follows the text."""

    def test_whitespace_only_changes_no_date(self):
        changed = page(BODY.replace('\n\n', '\n\n\n').replace('one inbox.', 'one   inbox.  '))
        self.repo.stage(PAGE, changed)
        self.repo.run_hook()
        self.assertUntouched(changed)

    def test_markdown_only_changes_no_date(self):
        # The #1179 fix: a blank line before `---`, plus bold and heading markup.
        changed = page(BODY.replace('{{< /blockquote >}}\n---', '{{< /blockquote >}}\n\n---')
                       .replace('**every**', 'every').replace('## Pricing', '### Pricing'))
        self.repo.stage(PAGE, changed)
        self.repo.run_hook()
        self.assertUntouched(changed)

    def test_front_matter_only_changes_no_date(self):
        changed = page(extra='[[lnks]]\nanchor = "help desk"')
        self.repo.stage(PAGE, changed)
        self.repo.run_hook()
        self.assertUntouched(changed)

    def test_text_edit_sets_lastmod_and_keeps_date(self):
        self.repo.stage(PAGE, page(BODY.replace('$15', '$19')))
        out = self.repo.run_hook()
        staged = self.repo.staged(PAGE)
        self.assertEqual(front_matter_value(staged, 'date'), PUBLISHED)
        self.assertFreshUtc(front_matter_value(staged, 'lastmod'))
        self.assertIn('(lastmod)', out)
        self.assertEqual(self.repo.read(PAGE), staged)

    def test_lastmod_goes_right_after_date(self):
        self.repo.stage(PAGE, page(BODY + '\nA new paragraph.\n'))
        self.repo.run_hook()
        lines = self.repo.staged(PAGE).splitlines()
        self.assertTrue(lines[2].startswith('date = '))
        self.assertTrue(lines[3].startswith('lastmod = '))

    def test_existing_lastmod_is_replaced(self):
        self.repo.stage(PAGE, page(lastmod='2026-07-07'))
        self.repo.commit()
        self.repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        self.repo.stage(PAGE, page(BODY + '\nA new paragraph.\n', lastmod='2026-07-07'))
        self.repo.run_hook()
        staged = self.repo.staged(PAGE)
        self.assertEqual(staged.count('lastmod = '), 1)
        self.assertFreshUtc(front_matter_value(staged, 'lastmod'))
        self.assertEqual(front_matter_value(staged, 'date'), PUBLISHED)

    def test_lastmod_set_by_hand_is_kept(self):
        changed = page(BODY + '\nRewritten.\n', lastmod='2026-08-25')
        self.repo.stage(PAGE, changed)
        self.repo.run_hook()
        self.assertUntouched(changed)

    def test_link_target_change_counts(self):
        repo = self.repo
        repo.stage(PAGE, page(BODY + '\nSee [pricing](/pricing/).\n'))
        repo.commit()
        repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        repo.stage(PAGE, page(BODY + '\nSee [pricing](/plans/).\n'))
        repo.run_hook()
        self.assertFreshUtc(front_matter_value(repo.staged(PAGE), 'lastmod'))

    def test_shortcode_removal_counts_unless_skipped(self):
        removed = page(BODY.replace(
            '{{< blockquote name="David Chandler" >}}\nSimply the best.\n{{< /blockquote >}}\n', ''))
        self.repo.stage(PAGE, removed)
        out = self.repo.run_hook(SKIP_CONTENT_DATES='1')
        self.assertIn('SKIP_CONTENT_DATES', out)
        self.assertUntouched(removed)

        self.repo.run_hook()
        self.assertFreshUtc(front_matter_value(self.repo.staged(PAGE), 'lastmod'))

    def test_skip_set_to_zero_does_not_skip(self):
        self.repo.stage(PAGE, page(BODY + '\nMore.\n'))
        self.repo.run_hook(SKIP_CONTENT_DATES='0')
        self.assertIsNotNone(front_matter_value(self.repo.staged(PAGE), 'lastmod'))

    def test_cjk_text_change_counts(self):
        repo = self.repo
        repo.stage(PAGE, page('\nカスタマーサポートのためのヘルプデスクです。\n'))
        repo.commit()
        repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        repo.stage(PAGE, page('\nカスタマーサポートのためのチャットです。\n'))
        repo.run_hook()
        self.assertFreshUtc(front_matter_value(repo.staged(PAGE), 'lastmod'))

    def test_published_page_without_date_gets_no_date(self):
        repo = self.repo
        repo.stage(PAGE, page(date=None))
        repo.commit()
        repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        repo.stage(PAGE, page(BODY + '\nMore.\n', date=None))
        repo.run_hook()
        staged = repo.staged(PAGE)
        self.assertIsNone(front_matter_value(staged, 'date'))
        self.assertIsNotNone(front_matter_value(staged, 'lastmod'))

    def test_body_already_on_main_is_not_redated(self):
        # A branch behind main picks up main's edit: the body equals main's, not HEAD's.
        repo = self.repo
        base = repo.git('rev-parse', 'HEAD').strip()
        repo.stage(PAGE, page(BODY + '\nEdited on main.\n'))
        repo.commit()
        repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        repo.git('checkout', '-q', '-b', 'feature', base)
        repo.stage(PAGE, page(BODY + '\nEdited on main.\n'))
        repo.run_hook()
        self.assertUntouched(page(BODY + '\nEdited on main.\n'))

    def test_merge_in_progress_is_skipped(self):
        changed = page(BODY + '\nMore.\n')
        self.repo.stage(PAGE, changed)
        git_dir = Path(self.repo.git('rev-parse', '--absolute-git-dir').strip())
        (git_dir / 'MERGE_HEAD').write_text(self.repo.git('rev-parse', 'HEAD'))
        self.repo.run_hook()
        self.assertUntouched(changed)

    def test_unstaged_changes_stay_unstaged(self):
        staged_version = page(BODY + '\nStaged paragraph.\n')
        self.repo.stage(PAGE, staged_version)
        disk_version = page(BODY + '\nStaged paragraph.\n\nNot staged yet.\n')
        self.repo.write(PAGE, disk_version)
        self.repo.run_hook()
        staged = self.repo.staged(PAGE)
        self.assertNotIn('Not staged yet.', staged)
        self.assertIsNotNone(front_matter_value(staged, 'lastmod'))
        disk = self.repo.read(PAGE)
        self.assertIn('Not staged yet.', disk)
        self.assertEqual(front_matter_value(disk, 'lastmod'), front_matter_value(staged, 'lastmod'))

    def test_translations_are_not_touched(self):
        de = 'content/de/page.md'
        changed = page(BODY + '\nMehr Text.\n')
        self.repo.stage(de, changed)
        self.repo.run_hook()
        self.assertEqual(self.repo.staged(de), changed)


class NewPage(HookTestCase):
    """The page is not on origin/main yet: date is the publish date and follows the work."""

    NEW = 'content/en/new-article.md'

    def test_new_page_gets_utc_date_and_no_lastmod(self):
        self.repo.stage(self.NEW, page(date='2026-10-14 15:27:00'))
        self.repo.run_hook()
        staged = self.repo.staged(self.NEW)
        self.assertFreshUtc(front_matter_value(staged, 'date'))
        self.assertIsNone(front_matter_value(staged, 'lastmod'))

    def test_new_page_without_date_gets_one_after_title(self):
        self.repo.stage(self.NEW, page(date=None))
        self.repo.run_hook()
        lines = self.repo.staged(self.NEW).splitlines()
        self.assertTrue(lines[1].startswith('title = '))
        self.assertFreshUtc(front_matter_value(lines[2], 'date'))

    def test_second_commit_on_the_branch_moves_date_not_lastmod(self):
        repo = self.repo
        repo.stage(self.NEW, page(date='2026-10-01T08:00:00Z'))
        repo.commit()
        repo.stage(self.NEW, page(BODY + '\nAnother section.\n', date='2026-10-01T08:00:00Z'))
        repo.run_hook()
        staged = repo.staged(self.NEW)
        self.assertFreshUtc(front_matter_value(staged, 'date'))
        self.assertIsNone(front_matter_value(staged, 'lastmod'))


class ManyPagesInOneCommit(HookTestCase):
    """Technical or editorial? From BULK_THRESHOLD pages on the hook asks the author."""

    files = {f'content/en/page-{i}.md': page() for i in range(hook.BULK_THRESHOLD)}

    CTA = '\n{{< cta heading="Get Started - 30 Days Free" primaryText="Start Free Trial" >}}\n'

    def change(self, count):
        paths = [f'content/en/page-{i}.md' for i in range(count)]
        for p in paths:
            self.repo.stage(p, page(BODY + self.CTA))
        return paths

    def lastmods(self, paths):
        return [front_matter_value(self.repo.staged(p), 'lastmod') for p in paths]

    def test_at_the_threshold_the_commit_stops_and_asks(self):
        paths = self.change(hook.BULK_THRESHOLD)
        before = {p: self.repo.staged(p) for p in paths}
        r = self.repo.run_hook_raw()
        self.assertEqual(r.returncode, 1)
        self.assertIn('SKIP_CONTENT_DATES=1', r.stderr)
        self.assertIn('FORCE_CONTENT_DATES=1', r.stderr)
        for p in paths:
            self.assertEqual(self.repo.staged(p), before[p])
            self.assertEqual(self.repo.read(p), before[p])

    def test_below_the_threshold_each_page_gets_lastmod(self):
        paths = self.change(hook.BULK_THRESHOLD - 1)
        self.repo.run_hook()
        for stamp in self.lastmods(paths):
            self.assertFreshUtc(stamp)

    def test_technical_change_with_skip_changes_no_dates(self):
        paths = self.change(hook.BULK_THRESHOLD)
        self.repo.run_hook(SKIP_CONTENT_DATES='1')
        self.assertEqual(self.lastmods(paths), [None] * len(paths))

    def test_text_edit_of_many_pages_with_force_sets_lastmod_on_each(self):
        paths = self.change(hook.BULK_THRESHOLD)
        self.repo.run_hook(FORCE_CONTENT_DATES='1')
        for stamp in self.lastmods(paths):
            self.assertFreshUtc(stamp)
        for p in paths:
            self.assertEqual(front_matter_value(self.repo.staged(p), 'date'), PUBLISHED)

    def test_a_new_page_in_a_forced_commit_gets_its_date(self):
        self.change(hook.BULK_THRESHOLD)
        new = 'content/en/new-article.md'
        self.repo.stage(new, page(date=None))
        self.repo.run_hook(FORCE_CONTENT_DATES='1')
        self.assertFreshUtc(front_matter_value(self.repo.staged(new), 'date'))

    def test_hand_set_lastmods_do_not_count_toward_the_question(self):
        # PAP refreshed 157 glossary articles with lastmod set by hand: nothing to ask.
        paths = [f'content/en/page-{i}.md' for i in range(hook.BULK_THRESHOLD)]
        for p in paths:
            self.repo.stage(p, page(BODY + self.CTA, lastmod='2026-10-08'))
        self.repo.run_hook()
        self.assertEqual(self.lastmods(paths), ['2026-10-08'] * len(paths))


class FileFormats(HookTestCase):
    def test_bom_file(self):
        self.repo.stage(PAGE, '﻿' + page(BODY + '\nMore.\n'))
        self.repo.run_hook()
        staged = self.repo.staged(PAGE)
        self.assertEqual(front_matter_value(staged, 'date'), PUBLISHED)
        self.assertIsNotNone(front_matter_value(staged, 'lastmod'))

    def test_crlf_file_keeps_its_line_endings(self):
        repo = self.repo
        repo.stage(PAGE, page(), newline='\r\n')
        repo.commit()
        repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        repo.stage(PAGE, page(BODY + '\nMore.\n'), newline='\r\n')
        repo.run_hook()
        for text in (repo.staged(PAGE), repo.read(PAGE)):
            self.assertIsNotNone(front_matter_value(text, 'lastmod'))
            self.assertNotIn('\n', text.replace('\r\n', ''))


class BodyWords(unittest.TestCase):
    def test_markup_and_whitespace_are_not_words(self):
        a = '+++\n+++\n## Title\n\nSome **bold** text.\n---\n'
        b = '+++\n+++\n### Title\nSome bold   text.\n\n---\n\n'
        self.assertEqual(hook.body_words(a), hook.body_words(b))

    def test_a_changed_word_is_a_change(self):
        self.assertNotEqual(hook.body_words('+++\n+++\nPlans at $15.'),
                            hook.body_words('+++\n+++\nPlans at $19.'))


if __name__ == '__main__':
    unittest.main()
