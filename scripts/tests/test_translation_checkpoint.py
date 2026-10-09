import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from translation_checkpoint import save_translation


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'translation.md'
        self.env = patch.dict(os.environ, {'TRANSLATION_CHECKPOINT_COMMAND': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_hook_sees_closed_complete_file_and_warnings(self):
        os.environ['TRANSLATION_CHECKPOINT_COMMAND'] = json.dumps(['publisher', 'checkpoint'])
        def publish(argv, **kwargs):
            self.assertEqual('completed', self.target.read_text())
            self.assertEqual(['warning'], json.loads(kwargs['input'])['warnings'])
            self.assertEqual(['publisher', 'checkpoint'], argv)
        with patch('translation_checkpoint.subprocess.run', side_effect=publish):
            save_translation(self.target, 'completed', ['warning'])
        self.assertEqual([self.target], list(Path(self.temp.name).iterdir()))

    def test_failure_before_atomic_replace_preserves_existing_target(self):
        self.target.write_text('previous complete file')
        with patch('translation_checkpoint.os.replace', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                save_translation(self.target, 'partial replacement')
        self.assertEqual('previous complete file', self.target.read_text())
        self.assertEqual([self.target], list(Path(self.temp.name).iterdir()))

    def test_failed_hook_aborts_instead_of_being_caught_as_translation_error(self):
        os.environ['TRANSLATION_CHECKPOINT_COMMAND'] = json.dumps(['publisher'])
        with patch('translation_checkpoint.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'publisher')):
            with self.assertRaises(SystemExit):
                save_translation(self.target, 'completed')
        self.assertEqual('completed', self.target.read_text())

    def test_no_hook_preserves_local_usage(self):
        with patch('translation_checkpoint.subprocess.run') as publish:
            save_translation(self.target, 'completed')
        publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
