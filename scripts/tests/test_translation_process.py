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


if __name__ == '__main__':
    unittest.main()
