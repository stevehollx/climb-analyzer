#!/usr/bin/env python3
"""
GitHub API client for cloud cache operations.

Handles file uploads, downloads, directory listing, and pull request creation
for the global-road-and-trail-climbs repository.
"""

import base64
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import requests


class GitHubClient:
    """Client for interacting with GitHub API for cloud cache operations."""

    def __init__(self, token: str, owner: str, repo: str):
        """
        Initialize GitHub client.

        Args:
            token: GitHub Personal Access Token (fine-grained, PR permissions only)
            owner: Repository owner (e.g., 'stevehollx')
            repo: Repository name (e.g., 'global-road-and-trail-climbs')
        """
        self.token = token
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        # Track if token is configured (not placeholder)
        self.token_configured = token and token != "YOUR_GITHUB_TOKEN_HERE"
        self.warned_401 = False  # Only warn once about missing token

    def list_directory_files(self, path: str, branch: str = "main") -> List[str]:
        """
        List all files in a directory.

        Args:
            path: Directory path in repo (e.g., "africa/algeria")
            branch: Branch to list from (default: main)

        Returns:
            List of filenames in the directory
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
        params = {"ref": branch}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            if response.status_code == 404:
                # Directory doesn't exist yet (common for new countries)
                return []
            elif response.status_code == 200:
                # Return only files, not directories
                items = response.json()
                if isinstance(items, list):
                    return [item['name'] for item in items if item['type'] == 'file']
                else:
                    # Single file returned instead of directory listing
                    return []
            else:
                if response.status_code == 401:
                    # Only warn once about missing token
                    if not self.warned_401 and not self.token_configured:
                        print(f"\n⚠️  Cloud cache unavailable: GitHub token not configured")
                        print(f"   To use cloud cache, set GITHUB_TOKEN in cloud_cache.py:33")
                        print(f"   (This is optional - cloud cache provides pre-computed analyses)\n")
                        self.warned_401 = True
                    elif not self.warned_401:
                        # Token is set but invalid
                        print(f"⚠️  Failed to list directory {path}: HTTP 401 (Unauthorized)")
                        print(f"    GitHub token may be invalid or expired")
                        self.warned_401 = True
                else:
                    print(f"⚠️  Failed to list directory {path}: HTTP {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error listing directory {path}: {e}")
            return []

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
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
        params = {"ref": branch}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=60)

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
        check_response = requests.get(url, headers=self.headers, params=params, timeout=30)

        data = {
            "message": message,
            "content": content_b64,
            "branch": branch
        }

        # If file exists, include SHA for update
        if check_response.status_code == 200:
            data["sha"] = check_response.json()['sha']

        try:
            response = requests.put(url, json=data, headers=self.headers, timeout=180)

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

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """
        Create a new branch.

        Args:
            branch_name: Name for new branch
            from_branch: Branch to create from (default: main)

        Returns:
            True if successful or already exists, False otherwise
        """
        # Get SHA of from_branch
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/ref/heads/{from_branch}"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)

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

            create_response = requests.post(create_url, json=data, headers=self.headers, timeout=30)

            if create_response.status_code in [201, 200]:
                return True
            elif create_response.status_code == 422:
                # Branch already exists - that's okay
                return True
            else:
                print(f"⚠️  Failed to create branch {branch_name}: HTTP {create_response.status_code}")
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
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls"

        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch
        }

        try:
            response = requests.post(url, json=data, headers=self.headers, timeout=30)

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
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/refs/heads/{branch_name}"

        try:
            response = requests.delete(url, headers=self.headers, timeout=30)
            return response.status_code == 204

        except requests.exceptions.RequestException:
            return False
