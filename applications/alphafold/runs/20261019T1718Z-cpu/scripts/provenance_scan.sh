#!/bin/bash
# Scans the run for forbidden outbound actions and secrets, for the review.
#
# Looks in RUN_DIR/scripts and RUN_DIR/logs for BigQuery export/load/write
# commands, GCS or paste uploads and git push; in RUN_DIR and in every commit
# for access-token patterns; and reports whether the current branch exists on
# any remote. Prints every hit; exit 0 when there is none, 1 otherwise.
#
# Usage: provenance_scan.sh RUN_DIR
set -uo pipefail
R=$(cd "$1" && pwd)
REPO=$(git -C "$R" rev-parse --show-toplevel)
hits=0
echo "scan $(date -u +%FT%TZ) run=$R"
echo "== BigQuery export/load/write, uploads, push (scripts/, logs/)"
grep -rIn -E "bq (extract|load|mk|cp|insert)|destination_table|--destination|create_table|create_dataset|gsutil|gcloud storage|storage\.googleapis|git push|ET_UPLOAD_GCS=|pastebin|transfer\.sh|0x0\.st" \
  "$R/scripts" "$R/logs" 2>/dev/null | grep -v -E "provenance_scan\.sh|unset .*ET_UPLOAD_GCS" && hits=1
echo "== token patterns (run dir)"
grep -rIl -E "ya29\.[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{20,}" "$R" 2>/dev/null && hits=1
echo "== token patterns (all commits)"
git -C "$REPO" log -p --all -G'ya29\.[A-Za-z0-9_-]{20,}' --oneline | head -5 | grep . && hits=1
echo "== branch on remotes"
br=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
git -C "$REPO" branch -r --list "*/$br" | grep . && hits=1
echo "branch $br: $(git -C "$REPO" branch -r --list "*/$br" | wc -l) remote refs; upstream: $(git -C "$REPO" rev-parse --abbrev-ref "$br@{upstream}" 2>&1)"
echo "result: $([ $hits -eq 0 ] && echo clean || echo HITS)"
exit $hits
