#!/bin/bash
# Validate the paper code release from a local checkout.
#
# This script is intentionally honest about what can be validated locally and
# what still depends on frozen result roots or a cluster environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_MAIN_REGEN=0

if [[ "${1:-}" == "--run-main-regen" ]]; then
    RUN_MAIN_REGEN=1
    shift
fi

MAIN_MANIFEST="${1:-experiments/report_sets/paper_public_main_v1.json}"
APPENDIX_MANIFEST="${2:-experiments/report_sets/paper_public_appendix_v1.json}"

cd "$REPO_ROOT"

step() {
    echo ""
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
}

warn() {
    echo "⚠️  $1"
}

require_file() {
    if [[ ! -e "$1" ]]; then
        echo "❌ Missing required path: $1"
        exit 1
    fi
}

step "Paper Local Release Validation"
echo "Repo root:        $REPO_ROOT"
echo "Main manifest:    $MAIN_MANIFEST"
echo "Appendix manifest:$APPENDIX_MANIFEST"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Branch:           $(git branch --show-current)"
else
    echo "Branch:           (not a git checkout)"
fi
echo "Run main regen:   $RUN_MAIN_REGEN"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
        warn "Working tree is not clean. Validation will continue, but this is not a pristine release state."
    else
        echo "✅ Working tree is clean."
    fi
else
    warn "Validation is running outside a git checkout; branch and cleanliness checks are skipped."
fi

step "1/7 Structural checks"
require_file "$MAIN_MANIFEST"
require_file "$APPENDIX_MANIFEST"
require_file "docs/release_manifest.md"
require_file "docs/release_artifact_inventory.json"
require_file "README.md"
require_file "docs/REPRODUCTION.md"

python -m json.tool "$MAIN_MANIFEST" >/dev/null
python -m json.tool "$APPENDIX_MANIFEST" >/dev/null
python -m json.tool "docs/release_artifact_inventory.json" >/dev/null
echo "✅ Canonical manifests and release inventory parse as JSON."

step "2/7 Checked-in bundle presence"
require_file "evaluation/sds/aggregated_report_batches/paper_public_main_v1"
require_file "evaluation/bigcode/aggregated_report_batches/paper_public_main_v1"
require_file "evaluation/sds/aggregated_report_batches/20260326_baseline-eval-v1"
echo "✅ Checked-in SDS, BigCode, and baseline bundles are present."

step "3/7 Shell and launcher syntax"
bash -n scripts/generate_paper_results.sh
bash -n scripts/generate_paper_appendix_results.sh
bash -n scripts/evaluate_baseline_evidence.sh
bash -n scripts/eval_capstor_sds_fixed_code.slurm
bash -n scripts/eval_capstor_sds_pipeline.slurm
bash -n scripts/train_capstor_unified_sds_qwen_coder.slurm
echo "✅ Local shell syntax checks passed."

step "4/7 Top-level evaluation tests"
PYTHONPATH=deps/open-r1/src:deps/syndeopt/src pytest --import-mode=importlib \
    tests/evaluation/test_reward_functions.py \
    tests/evaluation/test_fixed_code_pipeline.py \
    tests/evaluation/test_sds_evaluate.py \
    tests/evaluation/test_aggregate_plots.py \
    tests/evaluation/test_bigcode_aggregate.py

step "5/7 Unified open-r1 tests"
OPEN_R1_TESTS=(
    deps/open-r1/tests/test_sds_reward_normalization.py
    deps/open-r1/tests/test_feasibility_logging.py
)

if PYTHONPATH=deps/open-r1/src:deps/syndeopt/src python -c "import trl" >/dev/null 2>&1; then
    OPEN_R1_TESTS=(deps/open-r1/tests/test_rewards.py "${OPEN_R1_TESTS[@]}")
else
    warn "Skipping deps/open-r1/tests/test_rewards.py because the local environment does not provide the optional 'trl' dependency."
fi

PYTHONPATH=deps/open-r1/src:deps/syndeopt/src pytest --import-mode=importlib "${OPEN_R1_TESTS[@]}"

step "6/7 Appendix/supporting evidence validation"
if [ -d docs/technical-reports ]; then
    bash ./scripts/generate_paper_appendix_results.sh "$APPENDIX_MANIFEST"
else
    warn "Skipping appendix/supporting evidence validation because docs/technical-reports is not included in this bundle."
fi

step "7/7 Main bundle regeneration eligibility"
SDS_ROOTS_RAW="$(
python - "$MAIN_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

for path in data["sds"]["result_roots"]:
    print(path)
for path in data["bigcode"]["result_roots"]:
    print(path)
PY
)"

MISSING_ROOTS=()
while IFS= read -r root; do
    [[ -z "$root" ]] && continue
    if [[ ! -d "$root" ]]; then
        MISSING_ROOTS+=("$root")
    fi
done <<EOF
$SDS_ROOTS_RAW
EOF

if [[ ${#MISSING_ROOTS[@]} -eq 0 ]]; then
    echo "✅ Frozen local result roots are present."
    if [[ "$RUN_MAIN_REGEN" -eq 1 ]]; then
        echo "▶️  Running main paper regeneration..."
        bash ./scripts/generate_paper_results.sh "$MAIN_MANIFEST"

        PAPER_FACING_OUTPUTS=(
            evaluation/sds/aggregated_report_batches/paper_public_main_v1/final_results_table.tex
            evaluation/sds/aggregated_report_batches/paper_public_main_v1/error_types_table.tex
            evaluation/bigcode/aggregated_report_batches/paper_public_main_v1/bigcode_results_table.tex
        )

        DRIFTED_OUTPUTS=()
        for path in "${PAPER_FACING_OUTPUTS[@]}"; do
            if ! git diff --quiet -- "$path"; then
                DRIFTED_OUTPUTS+=("$path")
            fi
        done

        if [[ ${#DRIFTED_OUTPUTS[@]} -gt 0 ]]; then
            echo "❌ Main regeneration changed committed paper-facing outputs:"
            for path in "${DRIFTED_OUTPUTS[@]}"; do
                echo "  - $path"
            done
            echo "   Inspect the diffs above and reconcile them before treating the release as reproducible."
            exit 1
        fi
    else
        echo "ℹ️  Skipping main paper regeneration by default."
        echo "   Re-run with --run-main-regen to exercise the full local regeneration path."
    fi
else
    warn "Main paper regeneration skipped because some frozen result roots are not present locally:"
    for root in "${MISSING_ROOTS[@]}"; do
        echo "  - $root"
    done
fi

step "Validation summary"
echo "✅ Local structural checks passed."
echo "✅ Local test suite passed for the promoted SDS, BigCode, and open-r1 code paths."
echo "✅ Appendix/supporting evidence validation passed."
if [[ ${#MISSING_ROOTS[@]} -eq 0 ]]; then
    if [[ "$RUN_MAIN_REGEN" -eq 1 ]]; then
        echo "✅ Main paper regeneration was executed locally."
    else
        echo "ℹ️  Main paper regeneration is eligible locally but was not executed."
    fi
else
    warn "Main paper regeneration remains pending on a machine that has the frozen result roots synced."
fi
warn "Cluster-only validation still remains for sbatch/srun launchers, EDF environments, and Capstor checkpoint-path assumptions."
