#!/usr/bin/env python3
"""
GitHub App configuration for anonymous PR submissions.

SECURITY NOTICE:
This file contains a GitHub App private key with LIMITED permissions:
- Can ONLY create PRs to stevehollx/global-road-and-trail-climbs
- Cannot push directly to main branch
- Cannot delete or modify existing content without PR approval
- All actions require manual review before merging

This is intentionally embedded to allow users without GitHub accounts
to contribute climb analyses to the community repository.

If this key is ever abused:
1. Revoke it at: https://github.com/settings/apps/global-climbs-contributor
2. Generate a new key and update this file
"""

# GitHub App Configuration
GITHUB_APP_ID = "2188664"

# This private key has LIMITED permissions (PRs only to specific repo)
# It's safe to embed for this specific use case - see security notice above
GITHUB_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA8EAtfM6v49pl3DF6LJY0HkJYNwA3ZnDvJXK0chsH0tG0QYs9
u4vRVyPMQr7AWMwH9SY9pwQKJ3dF4DCXwutTjjggjj9gdasPBkvzr5iR80gj5TJD
fTVNl9LebiObR3gbBjqTPqpkIumff6/7RftIhsZ/sTlvmQBA9gvCwqOt3TUDUfbd
Cjo4lScCeS2Tyo5d42O7fbMZkouaJ3COfMsP4DAKGHpcr7GJFIbvqnn3mEvwZZqU
a6fsmyNP5Tvp1E/orq7GCGo7dYNVJP0CuE1wNPKJQ2By4TjcJmHJj6xQcS3Sx9tp
dIWAbKjc0zbeAiPxj4ri5QaQ6r04jlqnnSvowQIDAQABAoIBAFnputxdwggFQV/S
CIZNRH4amEclRpaJJ4cdUZjustPcdZieEuFwp0z0ccp89yGEYmoGAMbTxTUV90m7
BeEGD0RHjy+NWn1PIpVZsX6DHAQveHebgxSF8V8jpOkVXFS4B20iWN74B/fk9TNl
WiXLsE34VeuusixRov4yUpZjiXsiVVPVKDNTxIJUt7TqwiaNQZVO5OaMX9tCHgE6
ckm59lvKhtxd7odE+xWj7gEdOa9DSQ2tryCoCgpfcjvlALoKMQVIe/pb/uI0HKjk
kMwzhuanR1KQDA9emMe3mgqZ7uBZ6xwVIKAw1H8qkAZnYkfABguiAkbL+/s6ITn9
tufqr1ECgYEA+iu1mycugSYUHNVZ8WmAFH86IWBwAmEOaIo0ByRmAQzVBi/st2FS
HgUNgBac6AQWXJyUfuBuCfq12gMmPqt7N0wHoHLXK4RM4ZUdYD4hZ7VLqXv+H7no
8Q0QPkI7/pM1samnImpLwm5nSVouz4Y/UfLb9d8ITpms+hjXk/knTX0CgYEA9dlL
WZ3d0xzwXpUy7YE9oTIkIW570GZVkgWF42mX1dmZLJBBITJnyQ3LNO3PsUrILcUM
wXdnUqB+3SXiT0nTFbybtw2TDJ78flgH3Stq+FmYTb1ELMVGy2BKFXjBKdt0uib6
ynnMJDfGxvNlbFDDIbquF4GHSNkqwByzPH8wO5UCgYEA5De/olyqFfMw4eTX+l0u
FITD+PLK+8Cirkd1kxZnX4nfQ5ewsNG9YdlmKXV3iklARRgqd1wxxjTKdKnu17kD
3LwlMP/SvsYghKHNfKDxRHSlI2YTu8mTcWNjcAhoRqLwlrSX3dNubV7eJpJ4paRo
W0/bzX67S0jx/e1vrHPcoNkCgYEA8nzWLllgjzdO+uuMZOiB4jmzm2n/I4mcQz/B
VJI1Gc/bnjHWm6i853j8goNBxOwuz745G6XrOntlRjl0o6H8WsdCi2YPMXarMu8f
Ko4Fn1m3uI3C7anTbwvVZqJJXEDCPE3wNb+2k86T+G3gDtsF7IHV7wdqaXcSDJUC
UNxetZUCgYBE/ZINVFNmNx5WqQyikusvKoWbBt1ZGzM0tJ2pa+js68+ORUYsaVP4
AV8YEwWdin/aNmWw18XtqxILigAk1q1Cx9+z/lFosdyFM5LJAgdkYv7z2CR75YZQ
pdWvItKEUmFZYeXuPng/K7rT6Hu8Xawx7tPmilMotefjZ6h9q3nkyQ==
-----END RSA PRIVATE KEY-----"""

# Repository configuration
GITHUB_REPO_OWNER = "stevehollx"
GITHUB_REPO_NAME = "global-road-and-trail-climbs"

# Validation to ensure key is configured
def is_configured():
    """Check if the private key has been properly configured."""
    return (
        GITHUB_PRIVATE_KEY and
        "PASTE YOUR ACTUAL PRIVATE KEY HERE" not in GITHUB_PRIVATE_KEY and
        "BEGIN RSA PRIVATE KEY" in GITHUB_PRIVATE_KEY
    )

def get_config():
    """Get the GitHub App configuration."""
    return {
        'app_id': GITHUB_APP_ID,
        'private_key': GITHUB_PRIVATE_KEY,
        'owner': GITHUB_REPO_OWNER,
        'repo': GITHUB_REPO_NAME
    }