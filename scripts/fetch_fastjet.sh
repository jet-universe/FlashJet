#!/usr/bin/env bash
# Fetch the FastJet reference source into extern/ (kept out of git).
set -euo pipefail
VERSION="${1:-3.4.3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/extern"
cd "$ROOT/extern"
if [ ! -d "fastjet-$VERSION" ]; then
    curl -sL -O "https://fastjet.fr/repo/fastjet-$VERSION.tar.gz"
    tar -xzf "fastjet-$VERSION.tar.gz"
fi
echo "FastJet source at extern/fastjet-$VERSION"
