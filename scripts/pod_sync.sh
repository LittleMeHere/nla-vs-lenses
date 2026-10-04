#!/usr/bin/env bash
# Mirror the pod's /workspace/out and logs into runs/pod_sync/<pod>/ every INTERVAL seconds until
# the pod is unreachable. Usage: pod_sync.sh HOST PORT PODNAME [INTERVAL=300]
# Never stop or delete a pod before this has run once more by hand: bash pod_sync.sh H P NAME once
H=$1; PT=$2; NAME=$3; INT=${4:-300}; DEST=$(dirname "$0")/../runs/pod_sync/$NAME; mkdir -p "$DEST"
sync_once() {
  rsync -az --timeout=60 -e "ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p $PT" \
    --include='*/' --include='*.jsonl' --include='*.pt' --include='*.json' --include='*.log' --include='*.done' --exclude='*' \
    root@$H:/workspace/out/ root@$H:/workspace/ "$DEST"/ 2>>"$DEST/sync.err" && date -u +"synced %FT%TZ" >> "$DEST/sync.log"
}
if [ "$INT" = once ]; then sync_once; exit $?; fi
fails=0
while true; do
  if sync_once; then fails=0; else fails=$((fails+1)); [ $fails -ge 5 ] && { echo "pod unreachable 5x, exiting $(date -u)" >> "$DEST/sync.log"; exit 0; }; fi
  sleep "$INT"
done
