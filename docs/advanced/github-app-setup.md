# GitHub App Setup

The cloud cache uses a GitHub App for anonymous PR submissions. This page is for maintainers who need to configure or regenerate credentials.

!!! note "For Contributors"
    Regular users don't need to do anything. The embedded credentials work automatically.

## Why Embedded Credentials?

**Goal**: Allow users to contribute climb analyses without GitHub accounts.

**Security Model**:

- ✅ **Repository-scoped**: Only accesses the climbs repository
- ✅ **Permission-limited**: Contents + Pull Requests only
- ✅ **PR workflow**: Cannot push directly to main
- ✅ **Manual review**: All PRs require maintainer approval
- ✅ **Audit trail**: Actions attributed to bot account
- ✅ **Revocable**: Key can be regenerated anytime

**Worst case**: Someone spams PRs (close them, revoke key).

## Trade-offs

| Pros | Cons |
|------|------|
| Zero setup for contributors | Key is in source control |
| No OAuth flows | Anyone can create PRs |
| Works immediately | Key rotation needed if abused |

## Current Configuration

- **App Name**: Global Climbs Contributor
- **App ID**: 2188664
- **Repository**: stevehollx/global-road-and-trail-climbs
- **Permissions**:
  - Contents: Read & Write
  - Pull Requests: Read & Write

## Maintainer Setup

### Prerequisites

```bash
pip install PyJWT cryptography
```

### Run Setup Script

```bash
python3 setup_github_app.py
```

### Provide Private Key

Option 1: Paste key directly (from GitHub App settings)
Option 2: Provide path to `.pem` file

### Test Configuration

```bash
python3 test_cloud_cache_auth.py
```

### Commit Configuration

```bash
git add utils/cloud_cache_config.py
git commit -m "Configure GitHub App credentials"
git push
```

## Regenerating the Private Key

If compromised or for regular rotation:

1. Go to: https://github.com/settings/apps/global-climbs-contributor
2. Click "Generate a new private key"
3. Save the `.pem` file
4. Run `python3 setup_github_app.py`
5. Commit updated configuration

## How Authentication Works

```
1. App ID + Private Key → JWT token (10 min expiry)
2. JWT → Installation Access Token (1 hour expiry)
3. Installation token used for GitHub API calls
```

Tokens are auto-refreshed as needed.

## Troubleshooting

### "GitHub App authentication failed"

App not installed on repository:

1. Go to: https://github.com/settings/apps/global-climbs-contributor/installations
2. Click "Install App" or "Configure"
3. Select the repository

### "HTTP 401" on Installation Token

Private key is invalid or expired:

1. Regenerate key (see above)
2. Update configuration

### "ImportError: No module named 'jwt'"

```bash
pip install PyJWT cryptography
```

## Creating Your Own GitHub App

For forks or custom setups:

### 1. Create App

1. Go to GitHub Settings → Developer settings → GitHub Apps
2. Click "New GitHub App"
3. Configure:
   - Name: Your app name
   - Homepage: Your repo URL
   - Webhook: Disable (uncheck "Active")
   - Permissions:
     - Contents: Read & Write
     - Pull Requests: Read & Write
   - Where can this app be installed: Only this account

### 2. Install on Repository

1. Go to app settings → Install App
2. Select your repository

### 3. Generate Private Key

1. Go to app settings → General
2. Scroll to "Private keys"
3. Click "Generate a private key"
4. Save the `.pem` file

### 4. Configure Climb Analyzer

Update `utils/cloud_cache.py`:

```python
GITHUB_APP_ID = "your_app_id"
GITHUB_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
your_private_key_here
-----END RSA PRIVATE KEY-----"""
```

---

Next: [Parallel Processing](parallel-processing.md)
