"""
Agent-as-deployer: the last mile of the autonomous loop.

After a fix is verified, CI-gated, and merged, the agent deploys the healed app
onto THIS machine and exposes it through a single ngrok tunnel (a permanent domain
you start once). No per-repo host, no dashboards, no self-hosted runner.

Flow:
  merge -> clone/pull main -> pip install -> (re)start `python app.py` on DEPLOY_PORT
        -> health-check localhost:PORT -> read the public URL from ngrok's local API
        -> return the live URL

You run ONE tunnel, once:   ngrok http <DEPLOY_PORT> --domain=<your-permanent-domain>
It always points at DEPLOY_PORT; the agent just restarts whatever serves that port.
"""
import os
import sys
import time
import subprocess
import asyncio
import httpx

DEPLOY_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deployments")
DEPLOY_PORT = int(os.getenv("DEPLOY_PORT", "5000"))       # Flask's default port
NGROK_API = os.getenv("NGROK_API", "http://127.0.0.1:4040/api/tunnels")
# Optional override, e.g. "uvicorn main:app --host 0.0.0.0 --port {port}".
DEPLOY_START_CMD = os.getenv("DEPLOY_START_CMD", "")

# port -> running subprocess.Popen, so we can stop the old app before starting the new one.
_RUNNING: dict = {}


def _repo_dir(repo: str) -> str:
    return os.path.join(DEPLOY_ROOT, repo.replace("/", "__"))


async def _run(cmd: list, cwd: str = None, timeout: int = 300):
    """Run a build step (git/pip) with no shell and a timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 124, "", f"timed out after {timeout}s"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _stop_existing(port: int):
    """Kill the app currently serving `port` (and its child tree) if we started one."""
    p = _RUNNING.pop(port, None)
    if not p or p.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/F", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
    except Exception as e:  # noqa: BLE001
        print(f"[DEPLOY] Could not stop old process on {port}: {e}")


def _spawn_app(cmd: list, cwd: str, port: int, log_path: str) -> subprocess.Popen:
    """Start the app as a detached background process that survives this request."""
    env = os.environ.copy()
    env["PORT"] = str(port)          # apps that read PORT (the 12-factor way)
    env["FLASK_RUN_PORT"] = str(port)
    env["FLASK_APP"] = "app.py"
    log = open(log_path, "w", encoding="utf-8")
    kwargs = {}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=log, **kwargs)


async def _health_ok(port: int, attempts: int = 20, delay: float = 1.0) -> bool:
    url = f"http://127.0.0.1:{port}/"
    async with httpx.AsyncClient() as client:
        for _ in range(attempts):
            try:
                r = await client.get(url, timeout=3)
                if r.status_code < 500:      # any non-server-error means it's up
                    return True
            except Exception:                # not listening yet
                pass
            await asyncio.sleep(delay)
    return False


async def _public_url(port: int) -> str:
    """Ask the local ngrok agent for the public URL of the tunnel to our port."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(NGROK_API, timeout=5)
            tunnels = r.json().get("tunnels", [])
        https = [t for t in tunnels if t.get("proto") == "https"]
        pick = https[0] if https else (tunnels[0] if tunnels else None)
        return pick["public_url"] if pick else ""
    except Exception:                        # ngrok not running / API off
        return ""


async def deploy_repo(repo: str, token: str = None, port: int = None, provider: str = "github") -> dict:
    """
    Deploy `repo`'s current main branch locally and return where it's live.
    Returns {success, url, local_url, message}.
    """
    import shutil
    from agent.tools.test_runner import build_clone_cmd

    port = port or DEPLOY_PORT
    token = token or os.getenv("GITHUB_TOKEN", "") or os.getenv("GITLAB_TOKEN", "")
    os.makedirs(DEPLOY_ROOT, exist_ok=True)
    target = _repo_dir(repo)

    # 1. Always take a fresh shallow clone of main (provider-aware auth).
    shutil.rmtree(target, ignore_errors=True)
    code, _, err = await _run(build_clone_cmd(provider, repo, token, target))
    if code != 0:
        safe = err.replace(token, "********") if token else err
        return {"success": False, "url": "", "local_url": "", "message": f"checkout failed: {safe[:200]}"}

    # 2. Install dependencies (into the agent's interpreter; Flask deps are light).
    if os.path.isfile(os.path.join(target, "requirements.txt")):
        await _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=target)

    # 3. Decide how to start it.
    if DEPLOY_START_CMD:
        cmd = DEPLOY_START_CMD.format(port=port).split()
    else:
        entry = next((f for f in ("app.py", "main.py", "server.py", "wsgi.py")
                      if os.path.isfile(os.path.join(target, f))), None)
        if not entry:
            return {"success": False, "url": "", "local_url": "",
                    "message": "No app.py/main.py found and DEPLOY_START_CMD not set."}
        cmd = [sys.executable, entry]

    # 4. Restart the app on the fixed port.
    _stop_existing(port)
    log_path = os.path.join(target, "_deploy.log")
    try:
        proc = _spawn_app(cmd, target, port, log_path)
        _RUNNING[port] = proc
    except Exception as e:  # noqa: BLE001
        return {"success": False, "url": "", "local_url": "", "message": f"failed to start app: {e}"}

    # 5. Prove it's actually serving.
    local_url = f"http://127.0.0.1:{port}/"
    if not await _health_ok(port):
        tail = ""
        try:
            with open(log_path, encoding="utf-8") as f:
                tail = f.read()[-400:]
        except Exception:
            pass
        _stop_existing(port)
        return {"success": False, "url": "", "local_url": local_url,
                "message": f"app did not become healthy on port {port}. Log tail:\n{tail}"}

    # 6. Find the public link.
    public = await _public_url(port)
    if public:
        return {"success": True, "url": public, "local_url": local_url,
                "message": f"Live at {public} (via ngrok -> :{port})"}
    return {"success": True, "url": local_url, "local_url": local_url,
            "message": f"Running on {local_url}. Start ngrok to expose it: "
                       f"ngrok http {port} --domain=<your-permanent-domain>"}
