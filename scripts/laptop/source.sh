#!/usr/bin/env bash
# Shared setup step: fetch the audited upstream source and small robot assets.
ensure_sonic_source() {
 local revision=45c03facbf848e0c743091b7d9c17952096ff908
 if [[ ! -d "$UPSTREAM" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/NVlabs/GR00T-WholeBodyControl.git "$UPSTREAM"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$UPSTREAM" checkout --detach "$revision"
 fi
 if [[ ! -d "$UPSTREAM/gear_sonic_deploy" ]]; then
  echo "Incomplete upstream checkout at $UPSTREAM. Expected commit $revision; existing files were not changed." >&2
  return 1
 fi
 # Robot assets, the Unitree SDK/DDS libraries the controller links against, and the MuJoCo sim meshes.
 git -C "$UPSTREAM" lfs pull --exclude='' \
  --include='gear_sonic_deploy/g1/**,gear_sonic_deploy/thirdparty/unitree_sdk2/**,gear_sonic/data/robot_model/model_data/g1/**'
 # Fail before large downloads when an existing checkout isn't supported.
 /usr/bin/python3 - "$ROOT" "$UPSTREAM" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,upstream=map(Path,sys.argv[1:])
for name,digest in json.loads((root/'sonic_laptop/upstream-lock.json').read_text()).items():
    source=upstream/'gear_sonic_deploy'/name
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise SystemExit(f'Unsupported upstream file: {source}. Use the audited commit documented in docs/laptop-advanced.md; setup did not overwrite your source.')
PY
}
