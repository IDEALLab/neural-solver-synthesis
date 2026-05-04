#!/bin/bash
# Validate the appendix/supporting-evidence package for the paper release.
#
# Usage:
#   ./scripts/generate_paper_appendix_results.sh [appendix_manifest.json]
#
# Default: experiments/report_sets/paper_public_appendix_v1.json

set -euo pipefail

APPENDIX_MANIFEST="${1:-experiments/report_sets/paper_public_appendix_v1.json}"
SOFT_GATE_REPORT_SET="experiments/report_sets/paper_soft_gate_v1.json"

if [ ! -f "$APPENDIX_MANIFEST" ]; then
    echo "❌ Appendix manifest not found: $APPENDIX_MANIFEST"
    exit 1
fi

echo "📋 Validating appendix/supporting evidence from: $APPENDIX_MANIFEST"
echo ""

python - "$APPENDIX_MANIFEST" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)

checks = []
bundle = data["appendix_evidence"]["fixed_code_and_runtime_bundle"]
checks.append(bundle["path"])

soft_gate = data["appendix_evidence"]["soft_gate"]
checks.append(soft_gate["report"])

reward_norm = data["appendix_evidence"]["reward_normalization"]
checks.extend([reward_norm["report"], reward_norm["summary"]])

feas = data["appendix_evidence"]["feasibility_sparsity"]
checks.append(feas["report"])
checks.extend(os.path.join(feas["summary_dir"], name) for name in feas["summary_files"])

checks.append(data["appendix_evidence"]["timeouts"]["report"])

missing = [path for path in checks if not os.path.exists(path)]
if missing:
    print("Missing appendix/supporting artifacts:")
    for path in missing:
        print(f"  - {path}")
    sys.exit(1)

print("✅ Checked-in appendix/supporting artifacts are present.")
PY

echo ""
SOFT_GATE_ROOT="evaluation/sds/results_batches/20260326_soft-gate-v1"
if [ -f "$SOFT_GATE_REPORT_SET" ] && [ -d "$SOFT_GATE_ROOT" ]; then
    echo "📊 Regenerating SDS soft-gate aggregate from $SOFT_GATE_REPORT_SET ..."
    python evaluation/sds/aggregate_plots.py \
        --report-set "$SOFT_GATE_REPORT_SET" \
        --model-filter qwen2.5-coder-14b
    echo "✅ Soft-gate SDS aggregate updated."
else
    echo "ℹ️  Skipping soft-gate re-aggregation."
    echo "   Expected report set: $SOFT_GATE_REPORT_SET"
    echo "   Expected result root: $SOFT_GATE_ROOT"
fi
