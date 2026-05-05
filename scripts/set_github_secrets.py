"""
scripts/set_github_secrets.py
Sets all required GitHub Actions secrets for Muhammad-Subhan034/fraud-detection-mlops
using the GitHub REST API + libsodium encryption (PyNaCl).

Usage:
    python scripts/set_github_secrets.py --token ghp_your_token_here

Secrets set:
  KUBEFLOW_HOST     - Kubeflow UI URL (localhost for demo)
  KUBECONFIG        - Minimal kubeconfig pointing to localhost
  SLACK_WEBHOOK_URL - Placeholder (update with real URL for Slack alerts)
"""

import argparse
import base64
import json
import sys

import requests
from nacl import encoding, public

REPO  = "Muhammad-Subhan034/fraud-detection-mlops"
API   = f"https://api.github.com/repos/{REPO}"

# â”€â”€ Secret values (all pre-filled for this project) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SECRETS = {
    "KUBEFLOW_HOST": "http://localhost:8080",

    "KUBECONFIG": """apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: fraud-detection-demo
contexts:
- context:
    cluster: fraud-detection-demo
    user: demo-user
    namespace: fraud-detection
  name: fraud-detection-demo
current-context: fraud-detection-demo
users:
- name: demo-user
  user:
    token: demo-token-placeholder
""",

    "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/placeholder/not/configured",
}


def get_repo_public_key(headers: dict) -> tuple[str, str]:
    """Fetch the repo's public key for secret encryption."""
    resp = requests.get(f"{API}/actions/secrets/public-key", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["key_id"], data["key"]


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    """Encrypt secret value with repo public key using libsodium (NaCl)."""
    pub_key = public.PublicKey(public_key_b64.encode(), encoding.Base64Encoder)
    sealed  = public.SealedBox(pub_key)
    encrypted = sealed.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def set_secret(name: str, value: str, key_id: str,
               encrypted_value: str, headers: dict) -> bool:
    """Create or update a single GitHub Actions secret."""
    payload = {
        "encrypted_value": encrypted_value,
        "key_id":          key_id,
    }
    resp = requests.put(
        f"{API}/actions/secrets/{name}",
        headers=headers,
        json=payload,
    )
    # 201 = created, 204 = updated
    if resp.status_code in (201, 204):
        return True
    print(f"  ERROR {resp.status_code}: {resp.text}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Set GitHub Actions secrets for fraud-detection-mlops"
    )
    parser.add_argument(
        "--token", required=True,
        help="GitHub Personal Access Token (scopes: repo + workflow)"
    )
    args = parser.parse_args()

    headers = {
        "Authorization":        f"Bearer {args.token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    print()
    print("=" * 55)
    print(f"  Setting GitHub Secrets for {REPO}")
    print("=" * 55)
    print()

    # Verify token works
    me = requests.get("https://api.github.com/user", headers=headers)
    if me.status_code != 200:
        print(f"ERROR: Token invalid or expired ({me.status_code}).")
        print("Create a new token at: https://github.com/settings/tokens")
        print("Required scopes: repo, workflow")
        sys.exit(1)
    username = me.json().get("login", "unknown")
    print(f"  Authenticated as: {username}")
    print()

    # Get repo public key for encryption
    try:
        key_id, pub_key = get_repo_public_key(headers)
    except requests.HTTPError as e:
        print(f"ERROR fetching repo public key: {e}")
        sys.exit(1)

    # Set each secret
    all_ok = True
    for name, value in SECRETS.items():
        print(f"  Setting {name} ... ", end="", flush=True)
        encrypted = encrypt_secret(pub_key, value)
        ok = set_secret(name, value, key_id, encrypted, headers)
        if ok:
            print("âœ“")
        else:
            print("FAILED")
            all_ok = False

    print()

    # List all secrets to confirm
    resp = requests.get(f"{API}/actions/secrets", headers=headers)
    if resp.status_code == 200:
        secrets = resp.json().get("secrets", [])
        print("  Secrets now on repo:")
        for s in secrets:
            print(f"    â€¢ {s['name']}  (updated: {s['updated_at'][:10]})")

    print()
    print("=" * 55)
    if all_ok:
        print("  All secrets set successfully!")
        print()
        print("  Next: push any change to trigger the CI/CD pipeline:")
        print("    git add .")
        print("    git commit -m 'trigger: run CI/CD'")
        print("    git push")
        print()
        print("  Or trigger Stage 4 manually via GitHub Actions UI:")
        print("    https://github.com/Muhammad-Subhan034/fraud-detection-mlops/actions")
    else:
        print("  Some secrets failed â€” check errors above.")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()

