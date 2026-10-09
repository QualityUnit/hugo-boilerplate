"""Optional synchronous publication hook for completed translations."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def save_translation(target, text, warnings=None):
    """Atomically save, then publish; publication errors must abort the caller.

    TRANSLATION_CHECKPOINT_COMMAND is a JSON argv array, never a shell command.
    The hook receives a JSON object on stdin and must return only after its
    durable checkpoint succeeds. SystemExit intentionally bypasses the
    translator's per-file Exception handler: a failed push is not a reason to
    run another paid translation of the same page.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=target.parent, suffix='.tmp',
                                         delete=False) as output:
            temporary = output.name
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

    command = os.environ.get('TRANSLATION_CHECKPOINT_COMMAND')
    if command:
        try:
            argv = json.loads(command)
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
                raise ValueError('checkpoint command must be a nonempty JSON argv array')
            subprocess.run(argv, input=json.dumps({
                'path': str(target.resolve()), 'warnings': warnings or [],
            }), text=True, check=True, timeout=240)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f'[ERROR] Translation saved locally but checkpoint failed: {target}: {exc}',
                  file=sys.stderr, flush=True)
            raise SystemExit(1) from exc
