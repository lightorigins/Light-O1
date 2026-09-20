"""Lifecycle of a locally owned inference API process group."""

import os
import signal
import subprocess
import tempfile
from pathlib import Path


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 10) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A leader may exit before its GPU workers or encoder descendants finish.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class ManagedProcess:
    def __init__(self, temp_root: Path):
        self.temp_root = temp_root
        self.child: subprocess.Popen | None = None
        self.directory: tempfile.TemporaryDirectory | None = None

    @property
    def running(self):
        return self.child is not None and self.child.poll() is None

    def start(self, command):
        if self.running:
            raise RuntimeError("Inference service already running")
        self.stop()
        self.directory = tempfile.TemporaryDirectory(prefix="light-inference-", dir=self.temp_root)
        with (Path(self.directory.name) / "inference.log").open("wb") as log:
            self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(self):
        child = self.child
        if child is not None:
            terminate_process_group(child)
        self.child = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
