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

    def commit_file(
        self,
        branch_name: str,
        file_path: str,
        content: str,
        commit_message: str,
    ) -> bool:
        """
        Create or update a file in the repository on a specific branch.

        Uses GitHub Contents API: PUT /repos/{owner}/{repo}/contents/{path}

        Args:
            branch_name: Branch to commit to
            file_path: Path to file in repo (e.g., "releases/hawaii.md")
            content: File content (will be base64 encoded)
            commit_message: Commit message

        Returns:
            True if successful, False otherwise
        """
        import base64

        headers = self._get_headers()
        if not headers:
            return False

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{file_path}"

        # Check if file already exists to get its SHA (required for updates)
        existing_sha = None
        try:
            response = requests.get(
                url,
                headers=headers,
                params={"ref": branch_name},
                timeout=30
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")
        except requests.exceptions.RequestException:
            pass  # File doesn't exist, which is fine for creation

        # Prepare request data
        data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch_name,
        }

        if existing_sha:
            data["sha"] = existing_sha

        try:
            response = requests.put(url, headers=headers, json=data, timeout=60)

            if response.status_code in (200, 201):
                return True
            else:
                print(f"⚠️  Failed to commit file: HTTP {response.status_code}")
                print(f"   Response: {response.text[:300]}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error committing file: {e}")
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

    # ============================================================================
    # GitHub Releases API Methods
    # ============================================================================

    def create_release(
        self,
        tag_name: str,
        name: str,
        body: str,
        draft: bool = False,
        prerelease: bool = False,
        target_commitish: str = "main"
    ) -> Optional[Dict]:
        """
        Create a new GitHub release.

        Args:
            tag_name: Tag for the release (e.g., "california-v2.2.0")
            name: Release title
            body: Release description (markdown)
            draft: Whether this is a draft release
            prerelease: Whether this is a pre-release
            target_commitish: Branch or commit SHA

        Returns:
            Release data dict with 'id', 'upload_url', 'html_url' or None on failure
        """
        headers = self._get_headers()
        if not headers:
            return None

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases"

        data = {
            "tag_name": tag_name,
            "name": name,
            "body": body,
            "draft": draft,
            "prerelease": prerelease,
            "target_commitish": target_commitish,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)

            if response.status_code == 201:
                return response.json()
            else:
                print(f"⚠️  Failed to create release: HTTP {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error creating release: {e}")
            return None

    def get_release_by_tag(self, tag_name: str) -> Optional[Dict]:
        """
        Get an existing release by its tag name.

        Args:
            tag_name: Tag to look up (e.g., "california-v2.2.0")

        Returns:
            Release data dict or None if not found
        """
        headers = self._get_headers()
        if not headers:
            return None

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases/tags/{tag_name}"

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None  # Release doesn't exist
            else:
                print(f"⚠️  Failed to get release by tag: HTTP {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error getting release: {e}")
            return None

    def get_release_by_id(self, release_id: int) -> Optional[Dict]:
        """
        Get release info by ID (works for draft releases).

        Args:
            release_id: Release ID

        Returns:
            Release data dict or None if not found
        """
        headers = self._get_headers()
        if not headers:
            return None

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases/{release_id}"

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Failed to get release by ID: HTTP {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error getting release: {e}")
            return None

    def list_releases(self, per_page: int = 100) -> List[Dict]:
        """
        List all releases in the repository.

        Args:
            per_page: Number of releases per page (max 100)

        Returns:
            List of release data dicts
        """
        headers = self._get_headers()
        if not headers:
            return []

        releases = []
        page = 1

        while True:
            url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases"
            params = {"per_page": per_page, "page": page}

            try:
                response = requests.get(url, headers=headers, params=params, timeout=60)

                if response.status_code == 200:
                    page_releases = response.json()
                    if not page_releases:
                        break
                    releases.extend(page_releases)
                    if len(page_releases) < per_page:
                        break
                    page += 1
                else:
                    print(f"⚠️  Failed to list releases: HTTP {response.status_code}")
                    break

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Network error listing releases: {e}")
                break

        return releases

    def upload_release_asset(
        self,
        release_id: int,
        upload_url: str,
        local_file: Path,
        content_type: str = "application/octet-stream"
    ) -> Optional[Dict]:
        """
        Upload an asset to a GitHub release.

        Uses the uploads.github.com endpoint (different from api.github.com).
        Handles large files via streaming upload.

        Args:
            release_id: ID of the release
            upload_url: Upload URL from release creation (includes {?name,label})
            local_file: Path to the file to upload
            content_type: MIME type of the file

        Returns:
            Asset data dict with 'browser_download_url' or None on failure
        """
        headers = self._get_headers()
        if not headers:
            return None

        # The upload_url contains a template like:
        # https://uploads.github.com/repos/owner/repo/releases/123/assets{?name,label}
        # We need to strip the template part and add the filename as a query param
        base_url = upload_url.split("{")[0]
        url = f"{base_url}?name={local_file.name}"

        # Set content type for the file
        headers["Content-Type"] = content_type

        try:
            file_size = local_file.stat().st_size
            file_size_mb = file_size / (1024 ** 2)

            print(f"    Uploading {local_file.name} ({file_size_mb:.1f} MB)...")

            # Stream upload for large files
            with open(local_file, "rb") as f:
                response = requests.post(
                    url,
                    headers=headers,
                    data=f,
                    timeout=600  # 10 minute timeout for large files
                )

            if response.status_code == 201:
                asset_data = response.json()
                print(f"    ✓ Uploaded {local_file.name}")
                return asset_data
            else:
                print(f"⚠️  Failed to upload asset: HTTP {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error uploading asset: {e}")
            return None

    def list_release_assets(self, release_id: int) -> List[Dict]:
        """
        List all assets attached to a release.

        Args:
            release_id: ID of the release

        Returns:
            List of asset data dicts
        """
        headers = self._get_headers()
        if not headers:
            return []

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases/{release_id}/assets"

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Failed to list release assets: HTTP {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error listing assets: {e}")
            return []

    def delete_release_asset(self, asset_id: int) -> bool:
        """
        Delete a specific release asset.

        Args:
            asset_id: ID of the asset to delete

        Returns:
            True if successful, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases/assets/{asset_id}"

        try:
            response = requests.delete(url, headers=headers, timeout=30)
            return response.status_code == 204

        except requests.exceptions.RequestException:
            return False

    def delete_release(self, release_id: int) -> bool:
        """
        Delete a release.

        Args:
            release_id: ID of the release to delete

        Returns:
            True if successful, False otherwise
        """
        headers = self._get_headers()
        if not headers:
            return False

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases/{release_id}"

        try:
            response = requests.delete(url, headers=headers, timeout=30)
            return response.status_code == 204

        except requests.exceptions.RequestException:
            return False

    def update_release(
        self,
        release_id: int,
        name: Optional[str] = None,
        body: Optional[str] = None,
        draft: Optional[bool] = None,
        prerelease: Optional[bool] = None
    ) -> Optional[Dict]:
        """
        Update an existing release.

        Args:
            release_id: ID of the release to update
            name: New release title (optional)
            body: New release description (optional)
            draft: New draft status (optional)
            prerelease: New prerelease status (optional)

        Returns:
            Updated release data dict or None on failure
        """
        headers = self._get_headers()
        if not headers:
            return None

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/releases/{release_id}"

        data = {}
        if name is not None:
            data["name"] = name
        if body is not None:
            data["body"] = body
        if draft is not None:
            data["draft"] = draft
        if prerelease is not None:
            data["prerelease"] = prerelease

        if not data:
            return None  # Nothing to update

        try:
            response = requests.patch(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Failed to update release: HTTP {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error updating release: {e}")
            return None
