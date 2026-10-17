#!/usr/bin/env bash
# Stage DeepSeek-V4.1-Flash-1017 for jayleaton's two-Spark recipe (vendor/deepseek-v41-tensorfold-spark), on the head.
#
#   scripts/stage.sh setup     the recipe checkout with this repo's patch, config/prod.env from config/prod.env.example
#   scripts/stage.sh weights   the 1017 EXL3 2.9 bpw pack on both nodes, its chat template added
#   scripts/stage.sh engram    the 1017 Engram tables: download, pack each rank's half, rank 1's onto the worker
#   scripts/stage.sh build     the recipe's image on both nodes (no GPU needed; the CUDA prebuild waits for idle GPUs)
#   scripts/stage.sh bias      the 1017 image routing bias into both nodes' cache volume (not the older release's)
#   scripts/stage.sh all       every step above, in order
#
# Then, with the GPUs on both nodes idle: cd vendor/deepseek-v41-tensorfold-spark && scripts/serve.sh prebuild &&
# scripts/serve.sh preflight && scripts/serve.sh start.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
recipe=$here/vendor/deepseek-v41-tensorfold-spark
log() { printf '[stage] %s\n' "$*"; }

PACK_REPO=${PACK_REPO:-LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
BASE_REPO=${BASE_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
# The 1017 pack ships no chat template. DeepSeek's encoding/encoding.py is byte-identical in the 1017 and the earlier
# V4.1-Flash release, so the template of the pack the recipe was qualified on (Mia-AiLab's) applies unchanged.
TEMPLATE_REPO=${TEMPLATE_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}
ENGRAM_SRC=${ENGRAM_SRC:-}   # where the base repo's Engram shards (47, 48: ~190 GB) download; deletable after packing

env_file() { # the recipe's config: every path below comes from it
    [[ -f $recipe/config/prod.env ]] || { log "no $recipe/config/prod.env: run 'scripts/stage.sh setup' and edit it"; exit 1; }
    set -a; source "$recipe/config/prod.env"; set +a
    if grep -q '<' "$recipe/config/prod.env"; then log "config/prod.env still has <placeholders>"; exit 1; fi
}
wssh() { ssh -o BatchMode=yes "$WORKER_SSH" "$@"; }

setup() {
    git -C "$here" submodule update --init --recursive
    if git -C "$recipe" apply --check "$here/patches/0001-serve-nccl-cumem-off.patch" 2>/dev/null; then
        git -C "$recipe" apply "$here/patches/0001-serve-nccl-cumem-off.patch"; log "patch applied: NCCL cuMem off"
    else log "patch already applied (or the recipe moved on): check scripts/serve.sh"; fi
    [[ -f $recipe/config/prod.env ]] || { cp "$here/config/prod.env.example" "$recipe/config/prod.env"; \
        log "edit $recipe/config/prod.env: every <placeholder>, then run the other steps"; }
}

weights() {
    env_file
    log "pack $PACK_REPO -> $HEAD_MODEL"
    hf download "$PACK_REPO" --local-dir "$HEAD_MODEL" --max-workers 8
    hf download "$TEMPLATE_REPO" chat_template.jinja --local-dir "$HEAD_MODEL"
    python3 - "$HEAD_MODEL" <<'EOF'
import json, os, sys
d = sys.argv[1]; idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
miss = sorted({f for f in idx["weight_map"].values() if not os.path.exists(os.path.join(d, f))})
sys.exit(f"missing shards: {miss}") if miss else print(f"[stage] {len(set(idx['weight_map'].values()))} shards present")
EOF
    log "copying the pack to the worker ($WORKER_SSH:$WORKER_MODEL)"
    wssh "mkdir -p '$WORKER_MODEL'"
    rsync -a --info=progress2 --exclude .cache "$HEAD_MODEL/" "$WORKER_SSH:$WORKER_MODEL/"
}

engram() {
    env_file
    local src=${ENGRAM_SRC:-$(dirname "$HEAD_ENGRAM")/dsv41-1017-engram-src} r1; r1=$(mktemp -d "$(dirname "$HEAD_ENGRAM")/engram-r1.XXXX")
    mkdir -p "$src" "$HEAD_ENGRAM"
    hf download "$BASE_REPO" model.safetensors.index.json config.json --local-dir "$src"
    # shellcheck disable=SC2046
    hf download "$BASE_REPO" $(python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --list) --local-dir "$src" --max-workers 4
    log "packing rank 0 and rank 1 halves (~95 GB each)"
    python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --config "$HEAD_MODEL/config.json" --out "$HEAD_ENGRAM" --rank 0 &
    python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --config "$HEAD_MODEL/config.json" --out "$r1" --rank 1 &
    wait
    python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --config "$HEAD_MODEL/config.json" --check "$HEAD_ENGRAM" --rank 0
    wssh "mkdir -p '$WORKER_ENGRAM'"
    rsync -a --info=progress2 "$r1/" "$WORKER_SSH:$WORKER_ENGRAM/" && rm -rf "$r1"
    log "done; $src (the base shards) can be deleted"
}

build() { env_file; (cd "$recipe" && PREBUILD=0 scripts/serve.sh build); }

bias() { # the recipe's cache step fetches the older release's bias only when none is there, so this one stays
    env_file
    local f="docker run --rm -v $CACHE_VOL:/cache -e HF_HUB_OFFLINE=0 --entrypoint python $IMAGE"
    f+=" -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch $TF_DSV41_BIAS_VL --repo $BASE_REPO"
    bash -c "$f" && wssh "$f" && log "1017 image routing bias in $TF_DSV41_BIAS_VL on both nodes"
}

case "${1:-}" in
setup) setup ;; weights) weights ;; engram) engram ;; build) build ;; bias) bias ;;
all) setup; weights; engram; build; bias ;;
*) sed -n '2,12p' "$0"; exit 2 ;;
esac
