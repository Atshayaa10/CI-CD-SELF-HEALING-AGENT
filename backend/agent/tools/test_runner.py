"""
Verifier sandbox — empirically proves a patch before it ever reaches a PR.

Flow: clone the repo into a throwaway temp dir, write the proposed files, run pytest.
The tests run inside a Docker container (SANDBOX_MODE=docker) so LLM-generated code
never executes directly on the host. If Docker isn't available we transparently fall
back to running pytest in the temp dir on the host (SANDBOX_MODE=host, or auto-fallback).
"""
import os
import shutil
import tempfile
import asyncio
import subprocess
import traceback
from typing import Tuple, Dict

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.11-slim")
# Wall-clock cap so a hung install/test can't stall the whole pipeline.
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "300"))


async def _run_proc(cmd: list, cwd: str = None, timeout: int = SANDBOX_TIMEOUT):
    """Run a command with NO shell (RCE-safe) and a hard timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 124, "", f"Timed out after {timeout}s"
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _docker_available() -> bool:
    """True only if the Docker daemon is actually reachable."""
    if os.getenv("SANDBOX_MODE", "docker").lower() == "host":
        return False
    try:
        code, _, _ = await _run_proc(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15)
        return code == 0
    except Exception:  # noqa: BLE001 — docker not installed / not on PATH
        return False


async def extract_files_from_patch(patch: str) -> Dict[str, str]:
    """
    Parse the Solver's markdown code blocks into {file_path: content}.
    Handles both `# path/to/file.py` first-line comments and ```lang:path fences.
    """
    files_to_write = {}
    if not patch:
        return files_to_write

    in_block = False
    current_file = None
    current_content = []

    for line in patch.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_block:
                in_block = True
                current_file = None
                current_content = []
                if len(stripped) > 3:
                    parts = stripped[3:].split(":")
                    if len(parts) > 1:
                        current_file = parts[1].strip()
            else:
                in_block = False
                if current_file and current_file.strip():
                    files_to_write[current_file.strip()] = "\n".join(current_content)
        elif in_block:
            if current_file is None and (stripped.startswith("#") or stripped.startswith("//")):
                potential = stripped.lstrip("#/").strip()
                if potential and "." in potential and " " not in potential:
                    current_file = potential
                    continue  # the path comment itself is not file content
            current_content.append(line)

    return files_to_write


def _write_patch_files(test_dir: str, files_to_write: Dict[str, str]) -> Tuple[bool, str]:
    """Write patched files into the sandbox, blocking path-traversal escapes."""
    if not files_to_write:
        return False, "Error: No files were provided to the Verifier or patch parsing failed."

    for filepath, content in files_to_write.items():
        if not filepath or not filepath.strip():
            continue
        clean_path = filepath.strip().replace("\\", "/").lstrip("/")
        full_path = os.path.normpath(os.path.join(test_dir, clean_path))
        if not full_path.startswith(os.path.normpath(test_dir)):
            print(f"  [Sandbox] Security Block: path traversal blocked: {clean_path}")
            continue
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    return True, "ok"


def build_clone_cmd(provider: str, repo: str, token: str, target: str = ".") -> list:
    """Return an authenticated `git clone` command for the given SCM provider."""
    if provider == "gitlab":
        host = os.getenv("GITLAB_URL", "https://gitlab.com").split("://")[-1].rstrip("/")
        # GitLab's documented token-clone form.
        url = f"https://oauth2:{token}@{host}/{repo}.git"
        return ["git", "clone", "--depth", "50", url, target]
    # GitHub: pass the token via header so it never lands in the URL/args.
    url = f"https://github.com/{repo}.git"
    return ["git", "-c", f"http.extraHeader=Authorization: token {token}", "clone", "--depth", "50", url, target]


async def run_integration_tests(repo: str, commit_sha: str, files_to_write: Dict[str, str],
                                token: str = None, provider: str = "github") -> Tuple[bool, str]:
    """
    Clone -> apply patch -> run pytest (in Docker, or host fallback).
    `token` is the logged-in user's SCM token; falls back to env for dev.
    Returns (is_success, combined_output).
    """
    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GITLAB_TOKEN")
    if not token:
        return False, "Error: No SCM token available for cloning (please log in)."

    test_dir = tempfile.mkdtemp(prefix="opalite_sandbox_")
    use_docker = await _docker_available()
    mode = "Docker container" if use_docker else "host temp dir"

    try:
        print(f"  [Sandbox] Environment: {test_dir} (mode: {mode}, provider: {provider})")

        # 1. Clone securely
        code, _, stderr = await _run_proc(build_clone_cmd(provider, repo, token, "."), cwd=test_dir)
        if code != 0:
            return False, f"Git clone failed:\n{stderr.replace(token, '********')}"

        # 2. Checkout the target commit (best-effort; stay on default branch otherwise)
        target = commit_sha if commit_sha and len(commit_sha) > 5 else "HEAD"
        await _run_proc(["git", "checkout", target], cwd=test_dir)

        # 3. Apply the proposed patch
        ok, msg = _write_patch_files(test_dir, files_to_write)
        if not ok:
            return False, msg

        # 4. Run the tests
        if use_docker:
            is_success, output = await _run_tests_docker(test_dir)
        else:
            is_success, output = await _run_tests_host(test_dir)

        print(f"  [Sandbox] Result: {'PASSED' if is_success else 'FAILED'}")
        return is_success, f"[sandbox: {mode}]\n{output}"

    except Exception as e:  # noqa: BLE001
        return False, f"Exception in Verifier: {e}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def _run_tests_docker(test_dir: str) -> Tuple[bool, str]:
    """Run pytest inside an ephemeral container mounting the sandbox at /app."""
    # Install deps if present, then run pytest. `--rm` cleans the container up.
    # We keep default networking (pip needs it); the container is isolated from the host FS
    # except for the mounted sandbox dir.
    inner = (
        "python -m pip install -q --disable-pip-version-check pytest >/dev/null 2>&1; "
        "if [ -f requirements.txt ]; then python -m pip install -q -r requirements.txt >/dev/null 2>&1; fi; "
        "python -m pytest -v"
    )
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{test_dir}:/app",
        "-w", "/app",
        "--memory", "1g", "--cpus", "1",  # resource caps so a bad fix can't hog the box
        SANDBOX_IMAGE,
        "sh", "-c", inner,
    ]
    code, stdout, stderr = await _run_proc(cmd)
    return code == 0, f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


async def _run_tests_host(test_dir: str) -> Tuple[bool, str]:
    """Fallback: run pytest directly in the temp dir on this machine."""
    code, stdout, stderr = await _run_proc(["python", "-m", "pytest", "-v"], cwd=test_dir)
    return code == 0, f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
