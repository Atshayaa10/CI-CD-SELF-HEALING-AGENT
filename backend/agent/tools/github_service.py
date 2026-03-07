import os
import base64
import httpx
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)


class GithubService:
    """Service for interacting with the GitHub API — fetches files, creates branches, opens PRs."""

    def _headers(self):
        token = os.getenv("GITHUB_TOKEN", "")
        return {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}" if token else ""
        }

    def _token(self):
        return os.getenv("GITHUB_TOKEN", "")

    async def get_repo_files(self, repo_full_name: str, ref: str = "main") -> list:
        """Lists all files in the root of a repository."""
        url = f"https://api.github.com/repos/{repo_full_name}/contents/"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=self._headers(), params={"ref": ref})
            if r.status_code == 200:
                return [f["name"] for f in r.json() if f["type"] == "file"]
            return []

    async def get_file_content(self, repo_full_name: str, file_path: str, ref: str = "main") -> str:
        """Fetches the raw content of a file from the repository."""
        token = self._token()
        url = f"https://raw.githubusercontent.com/{repo_full_name}/{ref}/{file_path}"
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"token {token}"} if token else {}
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                return r.text
            return f"ERROR: Could not fetch {file_path} — HTTP {r.status_code}"

    async def get_failed_run_logs(self, repo_full_name: str, run_id: int) -> str:
        """Fetches the text logs for the first failed job in a GitHub Actions run."""
        url = f"https://api.github.com/repos/{repo_full_name}/actions/runs/{run_id}/jobs"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            data = r.json()
            failed_job = next(
                (j for j in data.get("jobs", []) if j.get("conclusion") == "failure"), None
            )
            if not failed_job:
                return "No failed jobs found in this run."
            job_id = failed_job["id"]
            log_url = f"https://api.github.com/repos/{repo_full_name}/actions/jobs/{job_id}/logs"
            log_r = await client.get(log_url, headers=self._headers(), follow_redirects=True)
            log_r.raise_for_status()
            return log_r.text

    async def create_fix_branch_and_pr(
        self, repo_full_name: str, base_branch: str,
        file_path: str, new_content: str, error_summary: str
    ) -> str:
        """Creates a new branch, commits the fixed file, and opens a Pull Request. Returns the PR URL."""
        import time
        branch_name = f"ai-fix/{int(time.time())}"
        headers = self._headers()

        async with httpx.AsyncClient() as client:
            # 1. Get HEAD SHA of base branch
            ref_url = f"https://api.github.com/repos/{repo_full_name}/git/ref/heads/{base_branch}"
            ref_r = await client.get(ref_url, headers=headers)
            ref_r.raise_for_status()
            base_sha = ref_r.json()["object"]["sha"]

            # 2. Create fix branch
            branch_url = f"https://api.github.com/repos/{repo_full_name}/git/refs"
            await client.post(branch_url, headers=headers, json={
                "ref": f"refs/heads/{branch_name}", "sha": base_sha
            })

            # 3. Get current file SHA
            file_url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
            file_r = await client.get(file_url, headers=headers, params={"ref": base_branch})
            file_sha = file_r.json().get("sha", "")

            # 4. Commit the fixed file
            encoded = base64.b64encode(new_content.encode()).decode()
            await client.put(file_url, headers=headers, json={
                "message": f"fix(ai): {error_summary[:80]}",
                "content": encoded, "sha": file_sha, "branch": branch_name
            })

            # 5. Open a Pull Request
            pr_url = f"https://api.github.com/repos/{repo_full_name}/pulls"
            pr_r = await client.post(pr_url, headers=headers, json={
                "title": f"🤖 AI Fix: {error_summary[:60]}",
                "body": f"**Opalite Auto-Healer** automatically detected and fixed this issue.\n\n**Root Cause:** {error_summary}\n**File:** `{file_path}`",
                "head": branch_name, "base": base_branch
            })
            return pr_r.json().get("html_url", f"PR creation failed: {pr_r.text[:100]}")


github_service = GithubService()
