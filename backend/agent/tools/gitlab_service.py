"""
GitLab service — parallel to github_service, exposing the SAME method names so the
heal flow can treat GitHub and GitLab interchangeably (pick the service, call uniformly).

Uses the GitLab v4 REST API. Auth is the logged-in user's OAuth token (Bearer),
bound per-request via a ContextVar; falls back to GITLAB_TOKEN for local/dev.
"""
import os
import time
import asyncio
import urllib.parse
import httpx
from contextvars import ContextVar

GITLAB_URL = os.getenv("GITLAB_URL", "https://gitlab.com").rstrip("/")
API = f"{GITLAB_URL}/api/v4"

_request_token: ContextVar[str] = ContextVar("gl_request_token", default="")


def set_request_token(token: str) -> None:
    _request_token.set(token or "")


class GitLabService:
    def _token(self):
        return _request_token.get() or os.getenv("GITLAB_TOKEN", "")

    def _headers(self):
        t = self._token()
        return {"Authorization": f"Bearer {t}"} if t else {}

    def _pid(self, repo_full_name: str) -> str:
        # GitLab accepts a URL-encoded "namespace/project" path as the project id.
        return urllib.parse.quote(repo_full_name, safe="")

    async def list_projects(self) -> list:
        """Projects the user is a member of, newest activity first."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{API}/projects", headers=self._headers(), params={
                "membership": "true", "per_page": 100,
                "order_by": "last_activity_at", "sort": "desc", "simple": "true",
            })
            if r.status_code == 200:
                return [{
                    "name": p["path_with_namespace"],
                    "private": p.get("visibility") != "public",
                    "updated_at": p.get("last_activity_at"),
                } for p in r.json()]
            return []

    async def get_repo_files(self, repo_full_name: str, ref: str = "main") -> list:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{API}/projects/{self._pid(repo_full_name)}/repository/tree",
                headers=self._headers(), params={"ref": ref, "per_page": 100},
            )
            if r.status_code == 200:
                return [f["name"] for f in r.json() if f["type"] == "blob"]
            return []

    async def get_file_content(self, repo_full_name: str, file_path: str, ref: str = "main") -> str:
        fp = urllib.parse.quote(file_path, safe="")
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{API}/projects/{self._pid(repo_full_name)}/repository/files/{fp}/raw",
                headers=self._headers(), params={"ref": ref},
            )
            if r.status_code == 200:
                return r.text
            return f"ERROR: Could not fetch {file_path} — HTTP {r.status_code}"

    async def file_exists(self, repo_full_name: str, file_path: str, ref: str = "main") -> bool:
        fp = urllib.parse.quote(file_path, safe="")
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{API}/projects/{self._pid(repo_full_name)}/repository/files/{fp}",
                headers=self._headers(), params={"ref": ref},
            )
            return r.status_code == 200

    async def create_fix_branch_and_pr(self, repo_full_name: str, base_branch: str,
                                       files_map: dict, error_summary: str) -> dict:
        """Create a branch + commit (multi-file) + open a Merge Request. Returns {pr_url, branch, head_sha}."""
        branch = f"ai-fix/{int(time.time())}"
        pid = self._pid(repo_full_name)

        # GitLab needs per-file action: 'update' if it exists on base, else 'create'.
        actions = []
        for path, content in files_map.items():
            exists = await self.file_exists(repo_full_name, path, base_branch)
            actions.append({"action": "update" if exists else "create", "file_path": path, "content": content})

        async with httpx.AsyncClient() as client:
            commit_r = await client.post(
                f"{API}/projects/{pid}/repository/commits", headers=self._headers(),
                json={"branch": branch, "start_branch": base_branch,
                      "commit_message": f"fix(ai): {error_summary[:80]}", "actions": actions},
            )
            commit_r.raise_for_status()
            head_sha = commit_r.json().get("id", "")

            mr_r = await client.post(
                f"{API}/projects/{pid}/merge_requests", headers=self._headers(),
                json={"source_branch": branch, "target_branch": base_branch,
                      "title": f"🤖 AI Fix: {error_summary[:60]}",
                      "description": f"**Opalite Auto-Healer** resolved a pipeline failure.\n\n"
                                     f"**Root Cause:** {error_summary}\n**Files:** {', '.join(files_map.keys())}"},
            )
            web_url = mr_r.json().get("web_url", f"MR creation failed: {mr_r.text[:100]}")
            return {"pr_url": web_url, "branch": branch, "head_sha": head_sha}

    async def wait_for_checks(self, repo_full_name: str, ref: str, timeout: int = 600, poll: int = 10) -> dict:
        """Poll the latest pipeline for `ref` until it settles. Returns {state, summary}."""
        pid = self._pid(repo_full_name)
        deadline = time.time() + max(timeout, 0)
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    r = await client.get(f"{API}/projects/{pid}/pipelines", headers=self._headers(),
                                         params={"ref": ref, "per_page": 1, "order_by": "id", "sort": "desc"})
                    r.raise_for_status()
                    pipelines = r.json()
                except Exception as e:  # noqa: BLE001
                    return {"state": "pending", "summary": f"Could not read pipelines: {e}"}

                if not pipelines:
                    return {"state": "none", "summary": "No pipeline configured on the fix branch."}

                status = pipelines[0].get("status")
                if status == "success":
                    return {"state": "success", "summary": "GitLab pipeline passed."}
                if status in ("failed", "canceled"):
                    return {"state": "failure", "summary": f"GitLab pipeline {status}."}
                if time.time() >= deadline:
                    return {"state": "pending", "summary": f"Pipeline still '{status}' after {timeout}s; not merging."}
                await asyncio.sleep(poll)

    async def merge_pull_request(self, mr_web_url: str) -> bool:
        """Merge an MR given its web URL (…/-/merge_requests/<iid>)."""
        try:
            after = mr_web_url.split(f"{GITLAB_URL}/", 1)[1]
            path, rest = after.split("/-/merge_requests/", 1)
            iid = rest.split("/")[0]
            pid = self._pid(path)
            async with httpx.AsyncClient() as client:
                r = await client.put(
                    f"{API}/projects/{pid}/merge_requests/{iid}/merge",
                    headers=self._headers(), json={"should_remove_source_branch": True},
                )
                return r.status_code == 200
        except Exception as e:  # noqa: BLE001
            print(f"GitLab merge error: {e}")
            return False

    async def trigger_deployment(self, webhook_url: str) -> bool:
        if not webhook_url:
            return False
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(webhook_url)
                return r.status_code in (200, 201, 202, 204)
        except Exception as e:  # noqa: BLE001
            print(f"Error deploying: {e}")
            return False

    async def rollback_to_previous_commit(self, repo_full_name: str, branch: str = "main") -> dict:
        """Revert the latest commit on `branch` using GitLab's revert API."""
        pid = self._pid(repo_full_name)
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{API}/projects/{pid}/repository/commits",
                                     headers=self._headers(), params={"ref_name": branch, "per_page": 1})
                r.raise_for_status()
                commits = r.json()
                if not commits:
                    return {"success": False, "message": "No commits found to roll back."}
                sha = commits[0]["id"]
                rev = await client.post(f"{API}/projects/{pid}/repository/commits/{sha}/revert",
                                        headers=self._headers(), json={"branch": branch})
                if rev.status_code in (200, 201):
                    body = rev.json()
                    return {"success": True, "message": f"Reverted {sha[:7]}",
                            "commit_url": body.get("web_url", "")}
                return {"success": False, "message": f"GitLab revert failed: HTTP {rev.status_code} {rev.text[:120]}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "message": f"Internal error: {e}"}


gitlab_service = GitLabService()
