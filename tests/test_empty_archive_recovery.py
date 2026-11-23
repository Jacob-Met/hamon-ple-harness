"""An empty output placeholder is recoverable only from observed new loose files."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import archive_checkpoint as checkpoint
import archive_contract
import delivery_guard


class EmptyArchiveRecovery(unittest.TestCase):
    names = ['first.json', 'notes.md']

    def test_startup_isolation(self):
        with tempfile.TemporaryDirectory(dir='/filesystem') as folder:
            root = Path(folder)
            target = root / 'result.zip'
            session = SimpleNamespace(instruction=(
                f'Build a first complete version of {target}, then improve. '
                'Produce exactly these two files at the archive root:\n1. `first.json`\n2. `notes.md`'))

            async def execute(command, timeout_sec):
                trial = subprocess.run(shlex.split(command), capture_output=True, text=True, check=True)
                return {'exit_code': 0, 'output': trial.stdout}

            session.exec = execute
            def snapshot():
                return asyncio.run(delivery_guard.snapshot(
                    session, [{'path': str(target), 'kind': 'file'}], observe_loose_baseline=True))

            before = snapshot()
            self.assertTrue(before[str(target)]['loose_members_initially_absent'])
            (root / 'notes.md').write_text('starter')
            inherited = snapshot()
            target.touch()
            empty = snapshot()
            for before_state in (inherited, None, empty):
                self.assertEqual(archive_contract.recovery_command('publish', str(target), before_state, empty),
                                 'publish')
            self.assertEqual(archive_contract.recovery_command('publish', str(target), before, empty),
                             'publish --after-empty')

    def test_no_mixed_or_invalid_publication_and_preserve_completed_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage, target = root / 'stage', root / 'result.zip'
            stage.mkdir()
            target.touch()
            (stage / 'first.json').write_text('{"old":true}')
            (root / 'notes.md').write_text('new output')
            with self.assertRaises(ValueError):
                checkpoint.publish_after_empty(stage, target, self.names)
            self.assertEqual(target.read_bytes(), b'')
            (root / 'first.json').write_text('invalid json')
            with self.assertRaises(ValueError):
                checkpoint.publish_after_empty(stage, target, self.names)
            self.assertEqual(target.read_bytes(), b'')
            (root / 'first.json').write_text('{"new":true}')
            checkpoint.publish_after_empty(stage, target, self.names)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(json.loads(archive.read('first.json')), {'new': True})
            saved = target.read_bytes()
            (stage / 'notes.md').write_text('other staged version')
            checkpoint.publish_after_empty(stage, target, self.names)
            self.assertEqual(target.read_bytes(), saved)

    def test_preserve_concurrent_publisher_on_either_side_of_placeholder_removal(self):
        for phase in ('before_park', 'before_link'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                target, other = root / 'result.zip', root / 'other.zip'
                target.touch()
                (root / 'first.json').write_text('{"ours":true}')
                (root / 'notes.md').write_text('ours')
                with zipfile.ZipFile(other, 'w') as archive:
                    archive.writestr('first.json', '{"other":true}')
                    archive.writestr('notes.md', 'independent writer')
                expected = other.read_bytes()
                replace, link = os.replace, os.link

                def interleave_replace(src, dst):
                    if phase == 'before_park' and src == target:
                        replace(other, target)
                    return replace(src, dst)

                def interleave_link(src, dst, **kwargs):
                    if phase == 'before_link' and other.exists() and dst == target:
                        replace(other, target)
                    return link(src, dst, **kwargs)

                with patch.object(os, 'replace', interleave_replace), patch.object(os, 'link', interleave_link):
                    checkpoint.publish_after_empty(root / 'stage', target, self.names)
                self.assertFalse(other.exists())
                self.assertEqual(target.read_bytes(), expected)
                self.assertEqual(list(root.glob('.ple-prior-*')), [])
