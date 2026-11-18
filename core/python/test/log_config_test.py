# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree.core.log_config."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from loguru import logger

from iree.core.log_config import configure_application_logging


class LogConfigTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp_path = Path(temp_dir.name)

    def configure(self, *args, **kwargs) -> int | None:
        """Configure logging, and remove the sink again after the test."""
        sink_id = configure_application_logging(*args, **kwargs)
        if sink_id is not None:
            # Cleanups run last in, first out: the file is closed before the
            # temporary directory is deleted.
            self.addCleanup(self.remove_sink, sink_id)
        return sink_id

    @staticmethod
    def remove_sink(sink_id: int) -> None:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass  # A later call of configure_application_logging replaced it.

    def test_file_sink_receives_messages_without_color_codes(self):
        log_file = self.tmp_path / "app.log"
        self.configure(log_file)

        logger.info("written to the file")

        content = log_file.read_text(encoding="utf-8")
        self.assertIn("written to the file", content)
        self.assertIn("INFO", content)
        self.assertNotIn("\x1b[", content)

    def test_level_filters_messages(self):
        log_file = self.tmp_path / "app.log"
        self.configure(log_file, level="WARNING")

        logger.info("too detailed")
        logger.warning("important")

        content = log_file.read_text(encoding="utf-8")
        self.assertNotIn("too detailed", content)
        self.assertIn("important", content)

    def test_format_is_applied(self):
        log_file = self.tmp_path / "app.log"
        self.configure(log_file, format="{level}:{message}")

        logger.info("short")

        self.assertEqual(log_file.read_text(encoding="utf-8"), "INFO:short\n")

    def test_sinks_of_the_application_are_kept(self):
        messages: list[str] = []
        application_sink = logger.add(
            messages.append, format="{message}", level="DEBUG"
        )
        self.addCleanup(self.remove_sink, application_sink)

        self.configure(self.tmp_path / "app.log")
        logger.info("to both sinks")

        self.assertEqual(messages, ["to both sinks\n"])
        self.assertIn(
            "to both sinks", (self.tmp_path / "app.log").read_text(encoding="utf-8")
        )

    def test_a_second_call_replaces_the_sink_of_the_first(self):
        first, second = self.tmp_path / "first.log", self.tmp_path / "second.log"
        self.configure(first)
        self.configure(second)

        logger.info("after reconfiguring")

        self.assertEqual(first.read_text(encoding="utf-8"), "")
        self.assertIn("after reconfiguring", second.read_text(encoding="utf-8"))

    def test_returned_id_removes_the_sink(self):
        log_file = self.tmp_path / "app.log"
        sink_id = self.configure(log_file)
        self.assertIsNotNone(sink_id)

        logger.remove(sink_id)
        logger.info("after removal")

        self.assertEqual(log_file.read_text(encoding="utf-8"), "")

    def test_failed_call_keeps_the_sink_in_place(self):
        log_file = self.tmp_path / "app.log"
        self.configure(log_file, format="{message}")

        with self.assertRaises(ValueError):
            self.configure(level="no-such-level")
        logger.info("still written")

        self.assertEqual(log_file.read_text(encoding="utf-8"), "still written\n")

    def test_braces_in_the_file_name_are_kept(self):
        log_file = self.tmp_path / "{run}" / "app.log"
        self.configure(log_file, format="{message}")

        logger.info("in the braces directory")

        self.assertEqual(
            log_file.read_text(encoding="utf-8"), "in the braces directory\n"
        )

    def test_nothing_is_added_without_stderr_and_without_a_file(self):
        # sys.stderr is None in applications without a console.
        with mock.patch.object(sys, "stderr", None):
            self.assertIsNone(self.configure())

    def run_python(self, script: str, **env: str) -> subprocess.CompletedProcess:
        """Run a script in a fresh interpreter, where Loguru is not set up yet."""
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**self.environment_without_loguru_settings(), **env},
        )

    @staticmethod
    def environment_without_loguru_settings() -> dict[str, str]:
        """The environment for a child that can import what this process can."""
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("LOGURU_")
        }
        environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        return environment

    def test_stderr_sink_replaces_the_preconfigured_one(self):
        result = self.run_python(
            "from loguru import logger\n"
            "from iree.core.log_config import configure_application_logging\n"
            "configure_application_logging(format='{message}')\n"
            "logger.info('exactly once')\n"
        )

        self.assertEqual(result.stderr, "exactly once\n")

    def test_first_application_sink_is_kept_without_a_preconfigured_one(self):
        # Without the preconfigured sink, the application's first sink gets its
        # id. It must not be mistaken for it.
        result = self.run_python(
            "import sys\n"
            "from loguru import logger\n"
            "from iree.core.log_config import configure_application_logging\n"
            "logger.add(sys.stdout, format='application sink: {message}', level=0)\n"
            "configure_application_logging(format='{message}')\n"
            "logger.info('kept')\n",
            LOGURU_AUTOINIT="0",
        )

        self.assertEqual(result.stdout, "application sink: kept\n")
        self.assertEqual(result.stderr, "kept\n")

    def test_application_sink_is_kept_when_loguru_was_imported_without_stderr(self):
        # Loguru then installs no sink of its own, whatever stderr is later.
        result = self.run_python(
            "import sys\n"
            "stderr, sys.stderr = sys.stderr, None\n"
            "from loguru import logger\n"
            "sys.stderr = stderr\n"
            "from iree.core.log_config import configure_application_logging\n"
            "logger.add(sys.stdout, format='application sink: {message}', level=0)\n"
            "configure_application_logging(format='{message}')\n"
            "logger.info('kept')\n"
        )

        self.assertEqual(result.stdout, "application sink: kept\n")
        self.assertEqual(result.stderr, "kept\n")

    def test_preconfigured_sink_is_replaced_whatever_the_environment_says_later(self):
        result = self.run_python(
            "import os\n"
            "from loguru import logger\n"
            "os.environ['LOGURU_AUTOINIT'] = '0'\n"
            "from iree.core.log_config import configure_application_logging\n"
            "configure_application_logging(format='{message}')\n"
            "logger.info('exactly once')\n"
        )

        self.assertEqual(result.stderr, "exactly once\n")


if __name__ == "__main__":
    unittest.main()
