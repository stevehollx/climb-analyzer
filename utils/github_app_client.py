#!/usr/bin/env python3
"""
GitHub App client for cloud cache operations.

Handles authentication via GitHub App (JWT + Installation tokens) instead of personal access tokens.
This allows public sharing of credentials without compromising security.
"""

import base64
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import requests


class GitHubAppClient:
    """Client for interacting with GitHub API using GitHub App authentication."""

    def __init__(self, app_id: str, private_key: str, owner: str, repo: str):
        """
        Initialize GitHub App client.

        Args:
            app_id: GitHub App ID (e.g., "2188664")
            private_key: Private key in PEM format (string or path to .pem file)
            owner: Repository owner (e.g., 'stevehollx')
            repo: Repository name (e.g., 'global-road-and-trail-climbs')
        """
        self.app_id = app_id
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"

        # Load private key
        # Check if it's a file path (short string) or the key content itself
        if len(private_key) < 500 and not private_key.startswith('-----BEGIN'):
            # Likely a file path - try to read it
            try:
                key_path = Path(private_key)
                if key_path.exists():
                    self.private_key = key_path.read_text()
                else:
                    # Path doesn't exist, treat as key content
                    self.private_key = private_key
            except (OSError, ValueError):
                # Not a valid path, treat as key content
                self.private_key = private_key
        else:
            # It's the key content directly
            self.private_key = private_key

        # Installation token cache
        self.installation_id = None
        self.installation_token = None
        self.token_expires_at = None

        # Track warnings
        self.warned_auth = False

    def _generate_jwt(self) -> str:
        """
        Generate JWT for GitHub App authentication.

        Returns:
            JWT token string
        """
        try:
            import jwt
        except ImportError:
            raise ImportError(
                "PyJWT library required for GitHub App authentication. "
                "Install with: pip install PyJWT cryptography"
            )

        # JWT payload
        now = int(time.time())
        payload = {
            'iat': now,  # Issued at
            'exp': now + 600,  # Expires in 10 minutes (max allowed)
            'iss': self.app_id  # Issuer (App ID)
        }

        # Generate JWT
        token = jwt.encode(payload, self.private_key, algorithm='RS256')
        return token

    def _get_installation_id(self) -> Optional[str]:
        """
        Get installation ID for the repository.

        Returns:
            Installation ID or None if not found
        """
        if self.installation_id:
            return self.installation_id

        jwt_token = self._generate_jwt()
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Get app installations
        url = f"{self.base_url}/app/installations"

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"⚠️  Failed to get app installations: HTTP {response.status_code}")
                return None

            installations = response.json()

            # Find installation for this repository
            for installation in installations:
                install_id = installation['id']

                # Get installation token to list repos
                token_url = f"{self.base_url}/app/installations/{install_id}/access_tokens"
                token_response = requests.post(token_url, headers=headers, timeout=30)

                if token_response.status_code == 201:
                    token_data = token_response.json()
                    install_token = token_data['token']

                    # Use installation token to list repos
                    install_headers = {
                        "Authorization": f"token {install_token}",
                        "Accept": "application/vnd.github.v3+json"
                    }

                    repos_url = installation['repositories_url']
                    repos_response = requests.get(repos_url, headers=install_headers, timeout=30)

                    if repos_response.status_code == 200:
                        repos = repos_response.json()
                        if 'repositories' in repos:
                            for repo_data in repos['repositories']:
                                if (repo_data['owner']['login'] == self.owner and
                                    repo_data['name'] == self.repo):
                                    self.installation_id = str(installation['id'])
                                    return self.installation_id

            print(f"⚠️  App not installed on {self.owner}/{self.repo}")
            return None

        except Exception as e:
            print(f"⚠️  Error getting installation ID: {e}")
            return None

    def _get_installation_token(self) -> Optional[str]:
        """
        Get or refresh installation access token.

        Returns:
            Installation access token or None if failed
        """
        # Check if we have a valid cached token
        if (self.installation_token and self.token_expires_at):
            from datetime import timezone
            now_utc = datetime.now(timezone.utc)
            if now_utc < self.token_expires_at - timedelta(minutes=5):
                return self.installation_token

        # Get installation ID
        installation_id = self._get_installation_id()
        if not installation_id:
            return None

        # Generate JWT
        jwt_token = self._generate_jwt()
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Get installation access token
        url = f"{self.base_url}/app/installations/{installation_id}/access_tokens"

        try:
            response = requests.post(url, headers=headers, timeout=30)

            if response.status_code != 201:
                print(f"⚠️  Failed to get installation token: HTTP {response.status_code}")
                return None

            data = response.json()
            self.installation_token = data['token']

            # Parse expiration time
            expires_at_str = data['expires_at']
            self.token_expires_at = datetime.fromisoformat(
                expires_at_str.replace('Z', '+00:00')
            )

            return self.installation_token

        except Exception as e:
            print(f"⚠️  Error getting installation token: {e}")
            return None

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """
        Get headers with valid installation token.

        Returns:
            Headers dict or None if authentication failed
        """
        token = self._get_installation_token()
        if not token:
            if not self.warned_auth:
                print("⚠️  GitHub App authentication failed")
                print("   Check that the app is installed on the repository")
                print("   and that the private key is correct")
                self.warned_auth = True
            return None

        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def list_directory_contents(self, path: str, branch: str = "main",
                                item_type: Optional[str] = None) -> List[str]:
        """
        List contents of a directory (files, directories, or both).

        Args:
            path: Directory path in repo (e.g., "africa/algeria")
            branch: Branch to list from (default: main)
            item_type: Filter by type - 'file', 'dir', or None for all (default: None)

        Returns:
            List of item names in the directory
        """
        headers = self._get_headers()
        if not headers:
            return []

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
        params = {"ref": branch}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 404:
                # Directory doesn't exist yet (common for new countries)
                return []
            elif response.status_code == 200:
                items = response.json()
                if isinstance(items, list):
                    if item_type:
                        return [item['name'] for item in items if item['type'] == item_type]
                    else:
                        return [item['name'] for item in items]
                else:
                    # Single file returned instead of directory listing
                    return []
            else:
                print(f"⚠️  Failed to list directory {path}: HTTP {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error listing directory {path}: {e}")
            return []

    def list_directory_files(self, path: str, branch: str = "main") -> List[str]:
        """
        List all files in a directory.

        Args:
            path: Directory path in repo (e.g., "africa/algeria")
            branch: Branch to list from (default: main)

        Returns:
            List of filenames in the directory
        """
        return self.list_directory_contents(path, branch, item_type='file')

    def download_file(self, path: str, output_path: Path, branch: str = "main") -> bool:
        """
        Download a file from GitHub.

        Args:
            path: File path in repo (e.g., "africa/algeria/file.xlsx")
            output_path: Local path to save file
            branch: Branch to download from (default: main)

        Returns:
            True if successful, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
        params = {"ref": branch}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()

                # GitHub returns base64-encoded content for files
                if 'content' in data:
                    content = base64.b64decode(data['content'])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(content)
                    return True
                # For large files, use download_url
                elif 'download_url' in data:
                    download_response = requests.get(data['download_url'], timeout=120)
                    if download_response.status_code == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(download_response.content)
                        return True

            print(f"⚠️  Failed to download {path}: HTTP {response.status_code}")
            return False

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error downloading {path}: {e}")
            return False

    def upload_file(self, local_path: Path, repo_path: str, branch: str,
                   message: Optional[str] = None) -> bool:
        """
        Upload a file to GitHub.

        Args:
            local_path: Local file to upload
            repo_path: Destination path in repo (e.g., "africa/algeria/file.xlsx")
            branch: Branch to upload to
            message: Commit message (auto-generated if None)

        Returns:
            True if successful, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        if not local_path.exists():
            print(f"⚠️  Local file not found: {local_path}")
            return False

        # Read file content
        content = local_path.read_bytes()
        content_b64 = base64.b64encode(content).decode('utf-8')

        # Auto-generate commit message if not provided
        if message is None:
            message = f"Add {local_path.name}"

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{repo_path}"

        # Check if file exists (to get SHA for update)
        params = {"ref": branch}
        check_response = requests.get(url, headers=headers, params=params, timeout=30)

        data = {
            "message": message,
            "content": content_b64,
            "branch": branch
        }

        # If file exists, include SHA for update
        if check_response.status_code == 200:
            data["sha"] = check_response.json()['sha']

        try:
            response = requests.put(url, json=data, headers=headers, timeout=180)

            if response.status_code in [201, 200]:
                return True
            else:
                print(f"⚠️  Failed to upload {repo_path}: HTTP {response.status_code}")
                if response.status_code == 422:
                    print(f"    Response: {response.json()}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error uploading {repo_path}: {e}")
            return False

    def delete_file(self, repo_path: str, branch: str, message: Optional[str] = None) -> bool:
        """
        Delete a file from GitHub.

        Args:
            repo_path: Path to file in repo (e.g., "africa/algeria/file.xlsx")
            branch: Branch to delete from
            message: Commit message (auto-generated if None)

        Returns:
            True if successful, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{repo_path}"

        # Get file SHA (required for deletion)
        params = {"ref": branch}
        try:
            check_response = requests.get(url, headers=headers, params=params, timeout=30)

            if check_response.status_code != 200:
                # File doesn't exist - consider this a success (already deleted)
                return True

            file_sha = check_response.json()['sha']

            # Auto-generate commit message if not provided
            if message is None:
                filename = repo_path.split('/')[-1]
                message = f"Remove {filename}"

            # Delete the file
            data = {
                "message": message,
                "sha": file_sha,
                "branch": branch
            }

            response = requests.delete(url, json=data, headers=headers, timeout=30)

            if response.status_code in [200, 204]:
                return True
            else:
                print(f"⚠️  Failed to delete {repo_path}: HTTP {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error deleting {repo_path}: {e}")
            return False

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """
        Create a new branch.

        Args:
            branch_name: Name for new branch
            from_branch: Branch to create from (default: main)

        Returns:
            True if successful or already exists, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        # Get SHA of from_branch
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/ref/heads/{from_branch}"

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"⚠️  Failed to get {from_branch} SHA: HTTP {response.status_code}")
                return False

            sha = response.json()['object']['sha']

            # Create new branch
            create_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/refs"
            data = {
                "ref": f"refs/heads/{branch_name}",
                "sha": sha
            }

            create_response = requests.post(create_url, json=data, headers=headers, timeout=30)

            if create_response.status_code in [201, 200]:
                return True
            elif create_response.status_code == 422:
                # Branch already exists - that's okay
                return True
            elif create_response.status_code == 403:
                print(f"⚠️  Failed to create branch {branch_name}: HTTP {create_response.status_code}")
                print(f"   GitHub App needs 'Contents: Read & write' permission")
                print(f"   Go to: https://github.com/settings/installations")
                print(f"   Click 'Configure' on your app → Repository access → Repository permissions")
                print(f"   Set 'Contents' to 'Read and write' and save")
                return False
            else:
                print(f"⚠️  Failed to create branch {branch_name}: HTTP {create_response.status_code}")
                try:
                    error_msg = create_response.json().get('message', 'Unknown error')
                    print(f"   Error: {error_msg}")
                except:
                    pass
                return False

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error creating branch: {e}")
            return False

    def create_pull_request(self, title: str, body: str, head_branch: str,
                           base_branch: str = "main") -> Optional[str]:
        """
        Create a pull request.

        Args:
            title: PR title
            body: PR description
            head_branch: Branch to merge from
            base_branch: Branch to merge into (default: main)

        Returns:
            PR URL if successful, None otherwise
        """
        headers = self._get_headers()
        if not headers:
            return None

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls"

        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch
        }

        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)

            if response.status_code == 201:
                pr_data = response.json()
                return pr_data['html_url']
            else:
                print(f"⚠️  Failed to create pull request: HTTP {response.status_code}")
                if response.status_code == 422:
                    print(f"    Response: {response.json()}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error creating pull request: {e}")
            return None

    def delete_branch(self, branch_name: str) -> bool:
        """
        Delete a branch (cleanup on failure).

        Args:
            branch_name: Name of branch to delete

        Returns:
            True if successful, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/refs/heads/{branch_name}"

        try:
            response = requests.delete(url, headers=headers, timeout=30)
            return response.status_code == 204

        except requests.exceptions.RequestException:
            return False

    def upload_files_via_git(
        self,
        local_files: List[Path],
        repo_path: str,
        branch_name: str,
        commit_message: str,
        delete_old_files: Optional[List[str]] = None
    ) -> bool:
        """
        Upload files using git CLI with LFS support for large files.

        This method handles files >100MB that exceed GitHub's API limit by using
        git push with LFS tracking.

        Args:
            local_files: List of local file paths to upload
            repo_path: Destination directory path in repo (e.g., "north-america/united-states-of-america/new-york")
            branch_name: Branch to push to
            commit_message: Commit message
            delete_old_files: Optional list of filenames to delete before adding new ones

        Returns:
            True if successful, False otherwise
        """
        import os
        import shutil
        import subprocess
        import tempfile

        # Get installation token for git authentication
        token = self._get_installation_token()
        if not token:
            print("⚠️  Failed to get authentication token for git push")
            return False

        # Create temporary directory for git operations
        temp_dir = tempfile.mkdtemp(prefix="cloud_cache_")

        try:
            repo_url = f"https://x-access-token:{token}@github.com/{self.owner}/{self.repo}.git"

            # Clone with sparse checkout (only the target directory)
            print(f"  Cloning repository (sparse checkout)...")

            # Initialize sparse checkout
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--sparse", "--depth=1",
                 "-b", "main", repo_url, temp_dir],
                check=True,
                capture_output=True,
                text=True
            )

            # Set sparse-checkout to include target directory
            subprocess.run(
                ["git", "-C", temp_dir, "sparse-checkout", "set", repo_path],
                check=True,
                capture_output=True,
                text=True
            )

            # Create branch or checkout existing
            try:
                subprocess.run(
                    ["git", "-C", temp_dir, "checkout", "-b", branch_name],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError:
                # Branch may already exist remotely
                subprocess.run(
                    ["git", "-C", temp_dir, "fetch", "origin", branch_name],
                    capture_output=True,
                    text=True
                )
                subprocess.run(
                    ["git", "-C", temp_dir, "checkout", branch_name],
                    check=True,
                    capture_output=True,
                    text=True
                )

            # Set git identity for commits (required in Docker containers)
            subprocess.run(
                ["git", "-C", temp_dir, "config", "user.email", "climb-analyzer@local"],
                check=True,
                capture_output=True,
                text=True
            )
            subprocess.run(
                ["git", "-C", temp_dir, "config", "user.name", "Climb Analyzer"],
                check=True,
                capture_output=True,
                text=True
            )

            # Set up Git LFS tracking for Excel files
            print(f"  Setting up Git LFS for large files...")
            subprocess.run(
                ["git", "-C", temp_dir, "lfs", "install", "--local"],
                check=True,
                capture_output=True,
                text=True
            )

            # Add LFS tracking pattern directly to .gitattributes
            # (Don't use `git lfs track` as it tries to update existing LFS pointer files)
            gitattributes_path = os.path.join(temp_dir, ".gitattributes")
            lfs_pattern = "*.xlsx filter=lfs diff=lfs merge=lfs -text"

            # Read existing .gitattributes if it exists
            existing_content = ""
            if os.path.exists(gitattributes_path):
                with open(gitattributes_path, "r") as f:
                    existing_content = f.read()

            # Add pattern if not already present
            if lfs_pattern not in existing_content:
                with open(gitattributes_path, "a") as f:
                    if existing_content and not existing_content.endswith("\n"):
                        f.write("\n")
                    f.write(lfs_pattern + "\n")

                # Stage .gitattributes
                subprocess.run(
                    ["git", "-C", temp_dir, "add", ".gitattributes"],
                    capture_output=True,
                    text=True
                )

            # Create target directory
            target_dir = os.path.join(temp_dir, repo_path)
            os.makedirs(target_dir, exist_ok=True)

            # Delete old files if specified
            if delete_old_files:
                for old_file in delete_old_files:
                    old_path = os.path.join(target_dir, old_file)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"  ✓ Removed old file: {old_file}")

            # Copy files to target directory
            print(f"  Copying files to repository...")
            for local_file in local_files:
                dest_path = os.path.join(target_dir, local_file.name)
                shutil.copy2(local_file, dest_path)
                file_size_mb = local_file.stat().st_size / (1024 * 1024)
                print(f"  ✓ Staged: {local_file.name} ({file_size_mb:.1f} MB)")

            # Stage all changes
            subprocess.run(
                ["git", "-C", temp_dir, "add", "-A"],
                check=True,
                capture_output=True,
                text=True
            )

            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "-C", temp_dir, "status", "--porcelain"],
                capture_output=True,
                text=True
            )

            if not status_result.stdout.strip():
                print("  ℹ️  No changes to commit")
                return True

            # Commit
            subprocess.run(
                ["git", "-C", temp_dir, "commit", "-m", commit_message],
                check=True,
                capture_output=True,
                text=True
            )

            # Push with LFS
            print(f"  Pushing to GitHub (with LFS)...")
            push_result = subprocess.run(
                ["git", "-C", temp_dir, "push", "-u", "origin", branch_name],
                capture_output=True,
                text=True
            )

            if push_result.returncode != 0:
                print(f"⚠️  Git push failed: {push_result.stderr}")
                return False

            print(f"  ✓ Successfully pushed to {branch_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git operation failed: {e}")
            if e.stderr:
                print(f"   Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"⚠️  Error during git upload: {e}")
            return False
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
