#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ReaDDy2 environment bootstrap for the LRZ CoolMUC-4 (cm4) Linux Cluster.
# Run this ON the cool.hpc.lrz.de LOGIN node (it needs internet; compute nodes
# have none). It is idempotent: safe to re-run.
#
#   bash bootstrap_login.sh
#
# Installs Miniforge (user-space conda) + a `readdy` env with ReaDDy2, then
# smoke-tests the import and the CPU kernel. Prints CONDA_BASE / ENV at the end
# for the SLURM submit scripts.
# ---------------------------------------------------------------------------
set -euo pipefail

MFDIR="${READDY_CONDA_BASE:-$HOME/miniforge3}"
ENVNAME="${READDY_ENV:-readdy}"

echo "=================================================================="
echo " ReaDDy2 bootstrap on $(hostname)"
echo " user=$USER  HOME=$HOME  SCRATCH=${SCRATCH:-<unset>}  WORK=${WORK:-<unset>}"
echo " target conda base: $MFDIR   env: $ENVNAME"
echo "=================================================================="

echo "-- home usage (env needs ~2-3 GB) --"
( quota -s 2>/dev/null || df -h "$HOME" 2>/dev/null | tail -1 ) || true

# 1. Miniforge -----------------------------------------------------------
if [ ! -x "$MFDIR/bin/conda" ]; then
  echo "-- installing Miniforge -> $MFDIR --"
  TMP="$(mktemp -d)"
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  ( cd "$TMP" && { curl -fsSLO "$URL" || wget -q "$URL"; } && bash "Miniforge3-Linux-x86_64.sh" -b -p "$MFDIR" )
  rm -rf "$TMP"
else
  echo "-- Miniforge already present at $MFDIR --"
fi

# shellcheck disable=SC1091
source "$MFDIR/bin/activate"
conda config --set channel_priority flexible >/dev/null 2>&1 || true

# 2. readdy env ----------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
  echo "-- env '$ENVNAME' already exists --"
else
  echo "-- creating env '$ENVNAME' (readdy + scientific stack); this solves for a few minutes --"
  conda create -y -n "$ENVNAME" -c readdy -c conda-forge \
      readdy numpy scipy pandas h5py
fi

conda activate "$ENVNAME"

# 3. smoke: import + CPU kernel -----------------------------------------
echo "-- smoke test: import readdy + build a CPU simulation --"
python - <<'PY'
import readdy, numpy as np, h5py
print("readdy    :", getattr(readdy, "__version__", "?"))
print("numpy     :", np.__version__)
print("h5py      :", h5py.__version__)
sys = readdy.ReactionDiffusionSystem(box_size=[30., 30., 30.])
sys.add_topology_species("Qt", 0.5)
sys.add_topology_species("Ft", 1.0)
sim = sys.simulation(kernel="CPU")
try:
    sim.kernel_configuration.n_threads = 4
    print("CPU kernel: OK, n_threads =", sim.kernel_configuration.n_threads)
except Exception as e:
    print("CPU kernel: threads note:", e)
print("SMOKE_OK")
PY

echo "=================================================================="
echo " DONE."
echo " CONDA_BASE=$MFDIR"
echo " ENV=$ENVNAME"
echo " -> use these in the SLURM submit (conda_base / conda_env)."
echo "=================================================================="
