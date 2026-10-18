#!/usr/bin/env bash
# The native engine's source tree: TensorFold main, jayleaton's DeepSeek-V4.1 series (ashhart/TensorFold PR #727),
# the series' newer kit engine (the recipe submodule's engine/zig, the files kit-overlay.txt lists), then this repo's
# patches (the kit on main's APIs, the 1017 pack's fp16 tensors, native images on main's server, video as frames).
#   engine/engine.sh DIR      writes the tree into DIR (a git repository; its last commit is the last patch)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
dir=${1:?usage: engine/engine.sh DIR}
MAIN=9ccf47276520e26bb78a63e01bddafd617878701      # ashhart/TensorFold main
SERIES=6e95bc6da90ad1b73a34c8c3028ae7d4d64ad007    # PR #727's head (jayleaton:stack/13b-performance-device)
KIT="$root/vendor/deepseek-v41-tensorfold-spark/engine/zig"   # the kit snapshot (engine 9131e10a on TensorFold 1.0.2)
[[ -f "$KIT/build.zig" ]] || { echo "no $KIT: git submodule update --init" >&2; exit 1; }
git clone -q https://github.com/ashhart/TensorFold.git "$dir"
cd "$dir"
git -c advice.detachedHead=false checkout -q "$MAIN"
git fetch -q origin "pull/727/head"
[[ "$(git rev-parse FETCH_HEAD)" == "$SERIES" ]] || echo "note: PR #727 moved on from $SERIES; merging the pinned commit" >&2
git fetch -q origin "$SERIES" 2>/dev/null || true
id=(-c user.name=recipe -c user.email=recipe@localhost)
git "${id[@]}" merge -q --no-edit "$SERIES"
rsync -a --files-from="$here/kit-overlay.txt" "$KIT/" ./
git add -A
git "${id[@]}" commit -q -m "The DeepSeek-V4.1 kit engine's family, kernels, server side and TP fabric over the series (deepseek-v41-tensorfold-spark engine/zig)"
git "${id[@]}" am -q "$here"/patches/*.patch
echo "engine tree at $(git rev-parse HEAD^{tree}) in $dir"
