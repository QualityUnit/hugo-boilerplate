"""Integration tests require the translator's normal requirements.txt packages."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

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

    def test_auth_failure_stops_batch_after_first_request(self):
        for status in (401, 403):
            with self.subTest(status=status):
                api = Mock()
                api.create_flow_session.side_effect = translator.flowhunt.exceptions.ApiException(status=status)
                with patch.object(translator, 'initialize_api_client'), \
                     patch.object(translator.flowhunt, 'FlowsApi', return_value=api), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, 'FLOWHUNT_WORKSPACE_ID'):
                        translator.process_translations(self.tasks, 'flow', 'workspace', 2)
                self.assertEqual(1, api.create_flow_session.call_count)
                api.invoke_flow_response.assert_not_called()
                self.assertFalse(self.tasks[0][3].exists())

    def test_attachment_auth_failure_stops_before_invoke(self):
        api = Mock()
        response = Mock(status_code=401)
        with patch.object(translator.requests, 'post', return_value=response) as upload, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, 'HTTP 401'):
                translator.create_translation_session(api, self.tasks[0][0], 'source', 'fr', 'flow', 'workspace')
        upload.assert_called_once()
        api.invoke_flow_response.assert_not_called()

    def test_poll_auth_failure_is_not_swallowed(self):
        api = Mock()
        api.poll_flow_response_without_preload_content.side_effect = translator.flowhunt.exceptions.ApiException(status=401)
        with self.assertRaisesRegex(SystemExit, 'HTTP 401'):
            translator.check_session_results(api, {'session_id': 'session', 'start_time': translator.time.time()})

    def test_rate_limit_still_retries(self):
        request = Mock(side_effect=[translator.flowhunt.exceptions.ApiException(status=429), 'ok'])
        with patch.object(translator.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual('ok', translator.retry_on_429(request))
        self.assertEqual(2, request.call_count)
        sleep.assert_called_once_with(2)


if __name__ == '__main__':
    unittest.main()
