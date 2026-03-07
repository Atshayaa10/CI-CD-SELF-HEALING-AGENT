import os
import shutil
import tempfile
import asyncio
import subprocess
import traceback
from typing import Tuple, Dict

async def extract_files_from_patch(patch: str) -> Dict[str, str]:
    """
    Parses the markdown code blocks from the Solver's patch to extract file paths and contents.
    Improved version to handle different comment styles and empty paths.
    """
    files_to_write = {}
    if not patch:
        return files_to_write
        
    lines = patch.split('\n')
    
    in_block = False
    current_file = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            if not in_block:
                in_block = True
                current_file = None
                current_content = []
                # Check if the path is on the same line as the fence: ```python:path/to/file.py
                if len(stripped) > 3:
                    parts = stripped[3:].split(':')
                    if len(parts) > 1:
                        current_file = parts[1].strip()
            else:
                in_block = False
                if current_file and current_file.strip():
                    files_to_write[current_file.strip()] = '\n'.join(current_content)
        elif in_block:
            # First line of the block might be the path as a comment: # path/to/file.py
            if current_file is None and (stripped.startswith('#') or stripped.startswith('//')):
                potential_file = stripped.lstrip('#/').strip()
                if potential_file and '.' in potential_file and ' ' not in potential_file:
                    current_file = potential_file
            
            if current_file is not None or not (stripped.startswith('#') or stripped.startswith('//')):
                current_content.append(line)
                
    return files_to_write

async def run_integration_tests(repo: str, commit_sha: str, files_to_write: Dict[str, str]) -> Tuple[bool, str]:
    """
    Clones the repository locally in a temp directory, applies the dict of file changes,
    and runs pytest. Returns (is_success, test_output).
    Hardened against RCE and Token Leakage.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return False, "Error: GITHUB_TOKEN not found for cloning."
        
    # Security: Use git -c http.extraHeader to pass the token securely instead of embedding it in the URL
    clone_url = f"https://github.com/{repo}.git"
    auth_header = f"Authorization: token {token}"
    
    test_dir = tempfile.mkdtemp(prefix="opalite_sandbox_")
    
    try:
        print(f"  [Sandbox] Created environment at: {test_dir}")
        
        async def run_proc(cmd: list, cwd: str):
            # Security: NO shell=True. Use direct list-style execution to prevent RCE.
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout.decode(), stderr.decode()

        # 1. Clone the repo securely
        print(f"  [Sandbox] Cloning {repo} securely...")
        # git -c http.extraHeader="Authorization: token <token>" clone <url> .
        retcode, stdout, stderr = await run_proc(
            ["git", "-c", f"http.extraHeader={auth_header}", "clone", clone_url, "."], 
            test_dir
        )
        if retcode != 0:
            # Redact token from logs just in case
            safe_stderr = stderr.replace(token, "********")
            return False, f"Git clone failed:\n{safe_stderr}"

        # 2. Checkout the commit
        checkout_target = commit_sha if commit_sha and len(commit_sha) > 5 else "main"
        print(f"  [Sandbox] Checking out {checkout_target}...")
        retcode, stdout, stderr = await run_proc(["git", "checkout", checkout_target], test_dir)
        
        # 3. Apply the patch
        if not files_to_write:
            return False, "Error: No files were provided to the Verifier or patch parsing failed."
            
        for filepath, content in files_to_write.items():
            if not filepath or filepath.strip() == "":
                continue
                
            clean_path = filepath.strip().replace('\\', '/')
            if clean_path.startswith('/'): clean_path = clean_path[1:]
            
            # Security: Path traversal check
            full_path = os.path.normpath(os.path.join(test_dir, clean_path))
            if not full_path.startswith(os.path.normpath(test_dir)):
                print(f"  [Sandbox] Security Block: Path traversal attempt blocked: {clean_path}")
                continue

            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
        # 4. Run tests securely
        print("  [Sandbox] Running tests via 'python -m pytest'...")
        # Security: Hard-coded safe entry point, no arbitrary shell commands.
        retcode, stdout, stderr = await run_proc(["python", "-m", "pytest", "-v"], test_dir)
        
        output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        is_success = (retcode == 0)
        
        print(f"  [Sandbox] Result: {'PASSED' if is_success else 'FAILED'}")
        return is_success, output
        
    except Exception as e:
        error_info = f"Exception in Verifier: {str(e)}\n{traceback.format_exc()}"
        print(f"  [Sandbox] {error_info}")
        return False, error_info
        
    finally:
        # Cleanup temp directory
        print(f"  [Sandbox] Cleaning up {test_dir}...")
        shutil.rmtree(test_dir, ignore_errors=True)
