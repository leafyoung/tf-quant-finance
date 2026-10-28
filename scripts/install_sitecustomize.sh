#!/usr/bin/env bash
# Reinstalls scripts/sitecustomize.py into .venv's site-packages, where Python
# actually auto-loads it. Needed after any `uv sync --reinstall` or fresh
# `.venv` creation, since .venv itself isn't tracked by git.
#
# Usage: bash scripts/install_sitecustomize.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_packages="$(cd "$repo_root" && uv run python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")"

cp "$repo_root/scripts/sitecustomize.py" "$site_packages/sitecustomize.py"
echo "Installed sitecustomize.py -> $site_packages/sitecustomize.py"
