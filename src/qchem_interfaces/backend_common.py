import os
import shutil
import subprocess


class QCBackendError(RuntimeError):
    """Error raised when a quantum-chemistry backend fails."""


class QCExecutableError(QCBackendError):
    """Error raised when a backend executable cannot be found or run."""


class QCParseError(QCBackendError):
    """Error raised when backend output cannot be parsed."""


def backend_scratch_dir(qcinput, backend_name, fallback_dir):
    scratch_dir = qcinput.get("scratch_dir")
    directory = os.path.join(scratch_dir, backend_name) if scratch_dir else fallback_dir
    os.makedirs(directory, exist_ok=True)
    return directory


def _tail(text, limit=6000):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "... output truncated ...\n" + text[-limit:]


def command_error(backend_name, returncode, stdout="", stderr="", command=None, cwd=None):
    message = f"{backend_name} command failed with exit code {returncode}."
    if command:
        message += f"\ncommand: {' '.join(str(part) for part in command)}"
    if cwd:
        message += f"\nworking directory: {cwd}"
    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()
    if stdout:
        message += f"\nstdout:\n{_tail(stdout)}"
    if stderr:
        message += f"\nstderr:\n{_tail(stderr)}"
    return QCBackendError(message)


def ensure_executable(path, backend_name):
    if not path:
        raise QCExecutableError(f"{backend_name} executable path is empty.")
    resolved = shutil.which(path) if os.path.basename(path) == path else path
    if resolved is None or not os.path.exists(resolved):
        raise QCExecutableError(f"{backend_name} executable was not found: {path}")
    if not os.access(resolved, os.X_OK):
        raise QCExecutableError(f"{backend_name} executable is not executable: {resolved}")
    return resolved


def run_backend_command(command, *, backend_name, cwd=None, check=True):
    if not command:
        raise QCExecutableError(f"{backend_name} command is empty.")
    command = list(command)
    command[0] = ensure_executable(command[0], backend_name)
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise QCExecutableError(f"{backend_name} executable was not found: {command[0]}") from exc
    except OSError as exc:
        raise QCExecutableError(f"{backend_name} command could not be started: {exc}") from exc
    if check and result.returncode != 0:
        raise command_error(
            backend_name,
            result.returncode,
            result.stdout,
            result.stderr,
            command=command,
            cwd=cwd,
        )
    return result


def parse_error(backend_name, detail, *, filename=None, stdout=None, stderr=None, cause=None):
    message = f"{backend_name} output parsing failed: {detail}"
    if filename:
        message += f"\nfile: {filename}"
    if stdout:
        message += f"\nstdout:\n{_tail(stdout)}"
    if stderr:
        message += f"\nstderr:\n{_tail(stderr)}"
    return QCParseError(message)


def cleanup_backend_scratch(qcinput, backend_name, fallback_dir):
    directory = backend_scratch_dir(qcinput, backend_name, fallback_dir)
    if not os.path.isdir(directory):
        return directory
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return directory


def should_retry_backend(qcinput):
    return bool(qcinput.get("backend_retry_after_cleanup", qcinput.get("retry_after_cleanup", False)))
