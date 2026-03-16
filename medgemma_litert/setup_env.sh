#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip

# LiteRT Torch nightly contains the recent HF export pipeline used below.
python -m pip install --upgrade litert-torch-nightly

# Runtime dependencies used by the conversion wrapper script.
python -m pip install --upgrade -r requirements.txt

echo "Environment ready at ${ROOT_DIR}/.venv"
