#!/usr/bin/env bash
# build.sh <src.py> <dest.ipynb> : expand @@GPU_SETUP@@, convert percent-format source to ipynb, execute it
set -euo pipefail
S=/tmp/claude-0/-home-user-ET-Miner/e480e379-ef5d-55e5-aace-dcfd7cc5503d/scratchpad
cd /home/user/ET-Miner
python3 -c "import sys; s=open(sys.argv[1]).read().replace('@@GPU_SETUP@@', open(sys.argv[2]).read().rstrip()); open(sys.argv[3],'w').write(s)" "$1" "$S/nbsrc/gpu_setup.txt" "$S/expanded.py"
uv run --with jupytext --with nbconvert --with ipykernel jupytext --quiet --to ipynb -o "$2" "$S/expanded.py"
python3 $S/fixmeta.py "$2"
uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 "$2" 2>&1 | grep -v "^\[NbConvertApp\] Writing\|IPKernelApp\|Converting" || true
test -s "$2"
