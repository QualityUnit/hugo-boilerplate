"""Integration tests require the translator's normal requirements.txt packages."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
with patch.dict(os.environ, {'FLOWHUNT_API_KEY': 'test-not-a-real-key'}):
    with contextlib.redirect_stdout(io.StringIO()):
        import translate_with_flowhunt as translator


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / 'content/en/page.txt'
        source.parent.mkdir(parents=True)
        source.write_text('English source')
        self.tasks = [(source, 'English source', 'fr', root / 'content/fr/page.txt'),
                      (source, 'English source', 'de', root / 'content/de/page.txt')]
        self.report = root / 'report.md'
        self.env = patch.dict(os.environ, {
            'TRANSLATION_REPORT_FILE': str(self.report),
            'GITHUB_STEP_SUMMARY': '', 'TRANSLATION_CHECKPOINT_COMMAND': '',
            'TRANSLATION_TIME_BUDGET_SECONDS': '0', 'TRANSLATION_DEADLINE_EPOCH': '',
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_process(self):
        with patch.object(translator, 'initialize_api_client') as client, \
             patch.object(translator.flowhunt, 'FlowsApi'), \
             patch.object(translator, 'create_translation_session', side_effect=[{'session_id': 'first'}, {'session_id': 'second'}]) as create, \
             patch.object(translator, 'check_session_results', return_value=(True, 'translated text')), \
             patch.object(translator.time, 'sleep'), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.created = create
            translator.process_translations(self.tasks, 'flow', 'workspace', 1)
            return create

    def test_complete_files_and_report(self):
        create = self.run_process()
        self.assertEqual(2, create.call_count)
        for task in self.tasks:
            self.assertEqual('translated text', task[3].read_text())
        self.assertIn('| Translated | **2** |', self.report.read_text())

    def test_expired_deadline_schedules_no_sessions_and_reports_pending(self):
        os.environ['TRANSLATION_DEADLINE_EPOCH'] = '1'
        with self.assertRaises(SystemExit) as result:
            self.run_process()
        self.assertEqual(2, result.exception.code)
        self.created.assert_not_called()
        self.assertIn('| Unfinished (resume on next run) | **2** |', self.report.read_text())

    def test_publication_failure_stops_before_next_paid_session(self):
        with patch.object(translator, 'save_translation', side_effect=SystemExit(1)), \
             patch.object(translator, 'create_translation_session', return_value={'session_id': 'first'}) as create, \
             patch.object(translator, 'initialize_api_client'), \
             patch.object(translator.flowhunt, 'FlowsApi'), \
             patch.object(translator, 'check_session_results', return_value=(True, 'translated text')), \
             patch.object(translator.time, 'sleep'), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                translator.process_translations(self.tasks, 'flow', 'workspace', 1)
        self.assertEqual(1, create.call_count)


PAGE_WITH_LINKS = '''+++
title = "Churn"
url = "/blog/churn/"

[[lnks_man]]
text = "help desk"
path = "/help-desk-software/"

[[lnks]]
text = "rules"
path = "/features/rules/"

[[lnks]]
text = "ticketing"
path = "/ticketing-software/"
+++
Body with [a link](/features/rules/).
'''

PAGE_WITHOUT_LINKS = '''+++
title = "Churn"
url = "/blog/churn/"

[[lnks_man]]
text = "help desk"
path = "/help-desk-software/"
+++
Body with [a link](/features/rules/).
'''


class StripGeneratedLinksTests(unittest.TestCase):
    def test_removes_every_lnks_table_and_keeps_the_rest(self):
        self.assertEqual(PAGE_WITHOUT_LINKS, translator.strip_generated_links(PAGE_WITH_LINKS))

    def test_table_between_other_tables_ends_at_the_next_header(self):
        text = ('+++\ntitle = "T"\n\n[[lnks]]\ntext = "a"\npath = "/a/"\n\n'
                '[[faq]]\nquestion = "Q"\n+++\nBody\n')
        self.assertEqual('+++\ntitle = "T"\n\n[[faq]]\nquestion = "Q"\n+++\nBody\n',
                         translator.strip_generated_links(text))

    def test_keeps_crlf_line_endings(self):
        text = PAGE_WITH_LINKS.replace('\n', '\r\n')
        self.assertEqual(PAGE_WITHOUT_LINKS.replace('\n', '\r\n'),
                         translator.strip_generated_links(text))

    def test_text_without_lnks_or_frontmatter_is_unchanged(self):
        for text in (PAGE_WITHOUT_LINKS, 'plain text\n[[lnks]]\n', ''):
            self.assertEqual(text, translator.strip_generated_links(text))

    def test_lnks_in_the_body_is_not_touched(self):
        text = '+++\ntitle = "T"\n+++\n[[lnks]]\npath = "/a/"\n'
        self.assertEqual(text, translator.strip_generated_links(text))


class TranslationWithoutGeneratedLinksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.content = Path(self.temp.name) / 'content'
        self.source = self.content / 'en/blog/churn.md'
        self.source.parent.mkdir(parents=True)
        self.source.write_text(PAGE_WITH_LINKS, encoding='utf-8')
        (self.content / 'es').mkdir()

    def test_source_is_sent_without_lnks(self):
        with contextlib.redirect_stdout(io.StringIO()):
            tasks, _ = translator.find_files_for_translation(self.content, ['es'])
        self.assertEqual([PAGE_WITHOUT_LINKS], [task[1] for task in tasks])

    def run_process(self, results):
        self.target = self.content / 'es/blog/churn.md'
        self.report = Path(self.temp.name) / 'report.md'
        tasks = [(self.source, PAGE_WITHOUT_LINKS, 'es', self.target)]
        env = {'TRANSLATION_REPORT_FILE': str(self.report),
               'GITHUB_STEP_SUMMARY': '', 'TRANSLATION_CHECKPOINT_COMMAND': '',
               'TRANSLATION_TIME_BUDGET_SECONDS': '0', 'TRANSLATION_DEADLINE_EPOCH': ''}
        # apply_url_policy() caches each language's URLs for the whole run;
        # keep this temp site's index out of the other tests.
        with patch.dict(os.environ, env), \
             patch.dict(translator._URL_INDEX_CACHE, clear=True), \
             patch.object(translator, 'initialize_api_client'), \
             patch.object(translator.flowhunt, 'FlowsApi'), \
             patch.object(translator, 'create_translation_session', side_effect=[
                 {'session_id': str(i)} for i in range(len(results))]) as create, \
             patch.object(translator, 'check_session_results', side_effect=results), \
             patch.object(translator.time, 'sleep'), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            translator.process_translations(tasks, 'flow', 'workspace', 1)
        return create

    def test_lnks_returned_by_the_flow_are_not_written(self):
        self.run_process([(True, PAGE_WITH_LINKS)])
        self.assertNotIn('[[lnks]]', self.target.read_text(encoding='utf-8'))
        self.assertIn('[[lnks_man]]', self.target.read_text(encoding='utf-8'))

    def test_mangled_lnks_row_does_not_fail_the_frontmatter_check(self):
        # An unquoted value is beyond repair_toml_frontmatter().
        mangled = PAGE_WITH_LINKS.replace('path = "/features/rules/"', 'path = /features/rules/')
        self.run_process([(True, mangled)])
        self.assertEqual(PAGE_WITHOUT_LINKS.strip(), self.target.read_text(encoding='utf-8'))
        self.assertIn('| Front matter still invalid | **0** |', self.report.read_text())

    def test_retry_sends_the_source_without_lnks(self):
        create = self.run_process([(True, None), (True, PAGE_WITHOUT_LINKS)])
        self.assertEqual(2, create.call_count)
        self.assertEqual(PAGE_WITHOUT_LINKS, create.call_args_list[1].args[2])


if __name__ == '__main__':
    unittest.main()
