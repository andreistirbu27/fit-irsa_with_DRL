#!/bin/bash

set -e
cd "$(dirname "$0")/.."

OUT_ROOT="figs/journal/policy_stats"
mkdir -p "$OUT_ROOT"

# Per-seed metrics: (config_label, source_dir_prefix)
# source_dir_prefix is the path prefix into which we append seed{N}.
declare -a CONFIGS=(
    "u8-s5    results/new/res-2p-var-users/res-2p-u8-s5-seed"
    "u15-s10  results/new/res-2p-var-load/res-2p-load-u15-s10-seed"
    "u20-s10  results/new/res-2p-var-load/res-2p-load-u20-s10-seed"
    "u25-s10  results/new/res-2p-var-load/res-2p-load-u25-s10-seed"
    "u30-s10  results/new/res-2p-var-load/res-2p-load-u30-s10-seed"
    "u20-s11  results/new/res-2p-var-users/res-2p-u20-s11-seed"
)

run_one() {
    local label="$1"
    local src_prefix="$2"
    local phase="$3"
    local metrics="$4"
    local seed="$5"
    local out="${OUT_ROOT}/res-2p-${label}-seed${seed}_p${phase}/"
    # Idempotency: skip if every requested metric already has an .npz
    local skip=1
    for m in $metrics; do
        # symbreak → symbreak.npz; fidelity → fidelity.npz; entropy → conditional_entropy.npz
        case "$m" in
            entropy) f="conditional_entropy.npz" ;;
            *)       f="${m}.npz" ;;
        esac
        if [ ! -f "${out}${f}" ]; then skip=0; break; fi
    done
    if [ $skip -eq 1 ]; then return 0; fi
    if [ ! -d "${src_prefix}${seed}" ]; then
        echo "  [skip] ${src_prefix}${seed} does not exist on disk yet"
        return 0
    fi
    python -m src.analysis.policy_stats \
        --result-dir "${src_prefix}${seed}" \
        --phase "$phase" --out "$out" \
        --metrics $metrics
}

echo "=== Per-seed metrics across (N, M) configs ==="
for line in "${CONFIGS[@]}"; do
    label=$(echo "$line" | awk '{print $1}')
    src=$(echo "$line"   | awk '{print $2}')
    echo ""
    echo "--- ${label} ---"
    for seed in $(seq 1 20); do
        run_one "$label" "$src" 1 "symbreak fidelity" "$seed"
        run_one "$label" "$src" 2 "symbreak entropy"  "$seed"
    done
done

echo ""
echo "=== Latent-D envelopes (Fig 10/11 reproduction at high-load) ==="
for line in "u30-s10  results/new/res-2p-var-load/res-2p-load-u30-s10-seed1   1" \
            "u20-s11  results/new/res-2p-var-users/res-2p-u20-s11-seed1       1"; do
    label=$(echo "$line" | awk '{print $1}')
    src=$(  echo "$line" | awk '{print $2}')
    phase=$(echo "$line" | awk '{print $3}')
    out="${OUT_ROOT}/2p_${label}_seed1_p${phase}/"
    if [ -f "${out}envelopes.npz" ] && [ -f "${out}latent_D.npz" ]; then
        echo "  [skip] ${out} already populated"
        continue
    fi
    if [ ! -d "$src" ]; then
        echo "  [skip] $src not on disk yet"; continue
    fi
    python -m src.analysis.policy_stats \
        --result-dir "$src" \
        --phase $phase --out "$out" \
        --metrics latent envelopes
done

echo ""
echo "=== Multi-seed TV distance (Fig 12 reproduction at multiple configs) ==="
declare -a TV_TARGETS=(
    "u8-s5    results/new/res-2p-var-users/res-2p-u8-s5-seed"
    "u15-s10  results/new/res-2p-var-load/res-2p-load-u15-s10-seed"
    "u20-s10  results/new/res-2p-var-load/res-2p-load-u20-s10-seed"
    "u20-s11  results/new/res-2p-var-users/res-2p-u20-s11-seed"
    "u30-s10  results/new/res-2p-var-load/res-2p-load-u30-s10-seed"
)
for line in "${TV_TARGETS[@]}"; do
    label=$(echo "$line" | awk '{print $1}')
    src=$(echo "$line"   | awk '{print $2}')
    out="${OUT_ROOT}/seed_distance_${label}/"
    if [ -f "${out}seed_tv_distance.npz" ]; then
        echo "  [skip] ${out} already populated"
        continue
    fi
    # Build the --result-dirs list, skipping seeds whose source dir is missing
    dirs=()
    for s in $(seq 1 20); do
        if [ -d "${src}${s}" ]; then dirs+=("${src}${s}"); fi
    done
    if [ ${#dirs[@]} -lt 2 ]; then
        echo "  [skip] ${label}: only ${#dirs[@]} seed dirs on disk; need ≥2"
        continue
    fi
    python -m src.analysis.policy_stats --multi-seed \
        --result-dirs "${dirs[@]}" \
        --out "$out"
done

echo ""
echo "=== Done. Run plot_policy_stats.py + the across-seed distribution helper to render figures. ==="
