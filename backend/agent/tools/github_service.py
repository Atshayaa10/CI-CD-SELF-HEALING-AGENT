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
        files_map: dict, error_summary: str
    ) -> str:
        """
        Creates a new branch, commits MULTIPLE files, and opens a Pull Request.
        Returns the PR URL.
        """
        import time
        branch_name = f"ai-fix/{int(time.time())}"
        headers = self._headers()

        async with httpx.AsyncClient() as client:
            # 1. Get HEAD SHA of base branch
            ref_url = f"https://api.github.com/repos/{repo_full_name}/git/ref/heads/{base_branch}"
            ref_r = await client.get(ref_url, headers=headers)
            ref_r.raise_for_status()
            base_sha = ref_r.json()["object"]["sha"]

            # 2. Get the tree SHA of the base commit
            commit_url = f"https://api.github.com/repos/{repo_full_name}/git/commits/{base_sha}"
            commit_r = await client.get(commit_url, headers=headers)
            commit_r.raise_for_status()
            base_tree_sha = commit_r.json()["tree"]["sha"]

            # 3. Create blobs/tree objects for each file
            tree_items = []
            for file_path, content in files_map.items():
                tree_items.append({
                    "path": file_path,
                    "mode": "100644",
                    "type": "blob",
                    "content": content
                })

            # 4. Create a new tree
            tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees"
            tree_r = await client.post(tree_url, headers=headers, json={
                "base_tree": base_tree_sha,
                "tree": tree_items
            })
            tree_r.raise_for_status()
            new_tree_sha = tree_r.json()["sha"]

            # 5. Create a new commit
            commit_payload = {
                "message": f"fix(ai): {error_summary[:80]}",
                "tree": new_tree_sha,
                "parents": [base_sha]
            }
            new_commit_r = await client.post(
                f"https://api.github.com/repos/{repo_full_name}/git/commits",
                headers=headers,
                json=commit_payload
            )
            new_commit_r.raise_for_status()
            new_commit_sha = new_commit_r.json()["sha"]

            # 6. Create fix branch pointing to the new commit
            branch_url = f"https://api.github.com/repos/{repo_full_name}/git/refs"
            await client.post(branch_url, headers=headers, json={
                "ref": f"refs/heads/{branch_name}", 
                "sha": new_commit_sha
            })

            # 7. Open a Pull Request
            pr_url = f"https://api.github.com/repos/{repo_full_name}/pulls"
            pr_r = await client.post(pr_url, headers=headers, json={
                "title": f"🤖 AI Fix: {error_summary[:60]}",
                "body": f"**Opalite Auto-Healer** detected and resolved multiple issues.\n\n**Root Cause:** {error_summary}\n**Files Impacted:** {', '.join(files_map.keys())}",
                "head": branch_name, 
                "base": base_branch
            })
            return pr_r.json().get("html_url", f"PR creation failed: {pr_r.text[:100]}")

    async def merge_pull_request(self, pr_url: str) -> bool:
        """Merges an open Pull Request using its HTML URL."""
        # Convert HTML URL (https://github.com/PDK45/repo/pull/1) 
        # to API URL (https://api.github.com/repos/PDK45/repo/pulls/1/merge)
        try:
            parts = pr_url.replace("https://github.com/", "").split("/")
            repo_full_name = f"{parts[0]}/{parts[1]}"
            pr_number = parts[3]
            
            api_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/merge"
            
            async with httpx.AsyncClient() as client:
                r = await client.put(
                    api_url, 
                    headers=self._headers(), 
                    json={"commit_title": "🚀 Auto-merged by Opalite AI", "merge_method": "squash"}
                )
                return r.status_code == 200
        except Exception as e:
            print(f"Error merging PR: {e}")
            return False

    async def trigger_deployment(self, webhook_url: str) -> bool:
        """Hits a remote webhook (Render/Vercel) to trigger a production deployment."""
        if not webhook_url:
            return False
            
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(webhook_url)
                return r.status_code in [200, 201, 202, 204]
        except Exception as e:
            print(f"Error deploying: {e}")
            return False


    async def rollback_to_previous_commit(self, repo_full_name: str, branch: str = "main") -> dict:
        """
        Safely rolls back a branch to its previous commit (HEAD~1) by creating a NEW commit
        that exactly matches the file tree of the previous commit. This advances history instead of rewriting it.
        """
        try:
            headers = self._headers()
            async with httpx.AsyncClient() as client:
                # 1. Get the current HEAD commit SHA
                ref_url = f"https://api.github.com/repos/{repo_full_name}/git/ref/heads/{branch}"
                r1 = await client.get(ref_url, headers=headers)
                r1.raise_for_status()
                current_sha = r1.json()["object"]["sha"]

                # 2. Get the commit details of HEAD to find its parent
                commit_url = f"https://api.github.com/repos/{repo_full_name}/git/commits/{current_sha}"
                r2 = await client.get(commit_url, headers=headers)
                r2.raise_for_status()
                commit_data = r2.json()
                
                if not commit_data["parents"]:
                    return {"success": False, "message": "Cannot rollback: No parent commits found (this is the first commit)."}
                
                parent_sha = commit_data["parents"][0]["sha"]

                # 3. Get the tree of the parent commit (this is what the repo looked like before the bad deploy)
                parent_commit_url = f"https://api.github.com/repos/{repo_full_name}/git/commits/{parent_sha}"
                r3 = await client.get(parent_commit_url, headers=headers)
                r3.raise_for_status()
                parent_tree_sha = r3.json()["tree"]["sha"]

                # 4. Create a NEW commit that uses the parent's file tree, but whose parent is the current HEAD
                new_commit_url = f"https://api.github.com/repos/{repo_full_name}/git/commits"
                payload = {
                    "message": f"🚨 AUTO-ROLLBACK: Reverting to stable commit {parent_sha[:7]}",
                    "tree": parent_tree_sha,
                    "parents": [current_sha] # The new commit continues from current HEAD
                }
                r4 = await client.post(new_commit_url, headers=headers, json=payload)
                r4.raise_for_status()
                new_commit_sha = r4.json()["sha"]
                new_commit_url_html = r4.json()["html_url"]

                # 5. Update the branch reference to point to this new rollback commit
                update_ref_url = f"https://api.github.com/repos/{repo_full_name}/git/refs/heads/{branch}"
                r5 = await client.patch(update_ref_url, headers=headers, json={"sha": new_commit_sha})
                r5.raise_for_status()

                return {
                    "success": True, 
                    "message": f"Successfully rolled back to {parent_sha[:7]}", 
                    "commit_url": new_commit_url_html
                }

        except httpx.HTTPStatusError as e:
            err_msg = e.response.json().get('message', str(e))
            print(f"Rollback HTTP Error: {err_msg}")
            return {"success": False, "message": f"GitHub API Error: {err_msg}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Internal Error: {str(e)}"}

github_service = GithubService()
