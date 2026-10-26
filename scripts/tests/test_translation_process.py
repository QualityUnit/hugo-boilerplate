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


class WorkspaceTests(unittest.TestCase):
    def test_cli_override_does_not_query_api(self):
        with patch.object(translator, 'DEFAULT_WORKSPACE_ID', 'env-workspace'), \
             patch.object(translator, 'initialize_api_client') as client:
            self.assertEqual('cli-workspace', translator.get_workspace_id('cli-workspace'))
        client.assert_not_called()

    def test_env_override_does_not_query_api(self):
        with patch.object(translator, 'DEFAULT_WORKSPACE_ID', 'env-workspace'), \
             patch.object(translator, 'initialize_api_client') as client:
            self.assertEqual('env-workspace', translator.get_workspace_id())
        client.assert_not_called()

    def test_default_workspace_comes_from_authenticated_key(self):
        api = Mock()
        api.get_user.return_value.api_key_workspace_id = 'key-workspace'
        with patch.object(translator, 'DEFAULT_WORKSPACE_ID', None), \
             patch.object(translator, 'initialize_api_client'), \
             patch.object(translator.flowhunt, 'WebAuthApi', return_value=api):
            self.assertEqual('key-workspace', translator.get_workspace_id())
        api.get_user.assert_called_once_with(_request_timeout=60)

    def test_missing_workspace_stops_without_hardcoded_fallback(self):
        api = Mock()
        api.get_user.return_value.api_key_workspace_id = None
        with patch.object(translator, 'DEFAULT_WORKSPACE_ID', None), \
             patch.object(translator, 'initialize_api_client'), \
             patch.object(translator.flowhunt, 'WebAuthApi', return_value=api):
            with self.assertRaisesRegex(SystemExit, 'did not return a workspace'):
                translator.get_workspace_id()

    def test_invalid_key_stops_workspace_discovery(self):
        api = Mock()
        api.get_user.side_effect = translator.flowhunt.exceptions.ApiException(status=401)
        with patch.object(translator, 'DEFAULT_WORKSPACE_ID', None), \
             patch.object(translator, 'initialize_api_client'), \
             patch.object(translator.flowhunt, 'WebAuthApi', return_value=api):
            with self.assertRaisesRegex(SystemExit, 'HTTP 401'):
                translator.get_workspace_id()
        api.get_user.assert_called_once()


class AttachmentTests(unittest.TestCase):
    def test_text_upload_preserves_filename_and_utf8_content(self):
        for suffix, mime in [('.md', 'text/markdown'), ('.MD', 'text/markdown'),
                             ('.markdown', 'text/markdown'), ('.html', 'text/html'),
                             ('.yaml', 'text/plain'), ('.yml', 'text/plain'),
                             ('.txt', 'text/plain')]:
            with self.subTest(suffix=suffix):
                api = Mock()
                api.create_flow_session.return_value.session_id = 'session'
                path = Path('page' + suffix)
                content = 'Príliš žltý kôň\n'
                with patch.object(translator.requests, 'post', return_value=Mock(status_code=200)) as upload, \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = translator.create_translation_session(api, path, content, 'fr', 'flow', 'workspace')
                self.assertEqual('session', result['session_id'])
                upload.assert_called_once_with(
                    'https://api.flowhunt.io/v2/flows/sessions/session/attachments',
                    headers={'Api-Key': translator.api_key},
                    files={'file': (path.name, content.encode('utf-8'), mime)}, timeout=60)
                api.invoke_flow_response.assert_called_once()

    def test_upload_error_reports_api_message_and_does_not_invoke(self):
        response = translator.requests.Response()
        response.status_code = 400
        response.reason = 'Bad Request'
        response._content = b'{"error_code":400,"message":"Error detecting file type..."}'
        self.assert_failed_upload_reports(response, 'Error detecting file type...')

    def test_upload_error_handles_non_json_response(self):
        response = translator.requests.Response()
        response.status_code = 502
        response.reason = 'Bad Gateway'
        response._content = b'<html>Bad Gateway</html>'
        self.assert_failed_upload_reports(response, 'Bad Gateway')

    def assert_failed_upload_reports(self, response, message):
        api = Mock()
        output = io.StringIO()
        with patch.object(translator.requests, 'post', return_value=response), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            result = translator.create_translation_session(api, Path('page.md'), 'source', 'fr', 'flow', 'workspace')
        self.assertIsNone(result)
        self.assertIn(f'HTTP {response.status_code}', output.getvalue())
        self.assertIn(message, output.getvalue())
        api.invoke_flow_response.assert_not_called()

    def test_upload_rate_limit_retries_before_invoking(self):
        api = Mock()
        limited = Mock(status_code=429, headers={'retry-after': '1'})
        with patch.object(translator.requests, 'post', side_effect=[limited, Mock(status_code=200)]) as upload, \
             patch.object(translator.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            result = translator.create_translation_session(api, Path('page.md'), 'source', 'fr', 'flow', 'workspace')
        self.assertIsNotNone(result)
        self.assertEqual(2, upload.call_count)
        self.assertEqual(upload.call_args_list[0], upload.call_args_list[1])
        sleep.assert_called_once_with(1)
        api.invoke_flow_response.assert_called_once()


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
