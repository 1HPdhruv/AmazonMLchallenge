#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu 24.04 EC2 instance (r6i.2xlarge or larger).
#   bash ec2_bootstrap.sh            # run as the default 'ubuntu' user, from the home directory
# Clones the code branch, installs Python 3.13 via uv (Ubuntu 24.04 ships 3.12), creates a clean venv
# pinned to requirements.txt, and runs the full test suite. Data is NOT fetched here (see docs/EC2_RUNBOOK.md).
set -euo pipefail

REPO=https://github.com/1HPdhruv/AmazonMLchallenge.git
BRANCH=venky/pipeline-v2
DIR=$HOME/AmazonMLchallenge

sudo apt-get update -y
sudo apt-get install -y git build-essential htop tmux unzip

# uv: standalone Python 3.13 builds, no system Python changes
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13

if [ ! -d "$DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$DIR"   # private repo: git will prompt for a GitHub token
fi
cd "$DIR"
git checkout "$BRANCH" && git pull --ff-only

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import numpy, pandas, scipy, sklearn, xgboost, rapidfuzz, jellyfish, yaml, psutil; \
print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'xgboost', xgboost.__version__, 'jellyfish', jellyfish.__version__)"

# Tests run on the synthetic fixtures they generate themselves; no real data needed.
unset PYTHONPATH
.venv/bin/python -m pytest -q
echo "Bootstrap OK. Next: copy Dataset/ (docs/EC2_RUNBOOK.md step 4)."
