#!/usr/bin/env bash
# Stage DeepSeek-V4.1-Flash-1017 on native TensorFold for two DGX Sparks (TP=2), from the head node.
#
#   scripts/stage.sh setup     the submodule; config/native.env from config/native.env.example (then edit it)
#   scripts/stage.sh weights   the 1017 EXL3 2.9 bpw pack with a chat template, on both nodes (~211 GB)
#   scripts/stage.sh engram    the 1017 Engram tables: download (~190 GB), each rank's half, rank 1's onto the worker
#   scripts/stage.sh engine    the engine tree (engine/engine.sh), its image (docker/Dockerfile), onto the worker
#   scripts/stage.sh pyimage   jayleaton's Python image: the asset tools' Torch, Triton 3.7.1 and reference source
#   scripts/stage.sh assets    RoPE and Engram host tables, the Triton AOT set (4K and 2K prefill), onto the worker
#   scripts/stage.sh cache     the 1017 image routing bias and the dense-prefill table, on both nodes
#   scripts/stage.sh all       every step above, in order
#   scripts/stage.sh start     rank 1 on the worker, then rank 0 here (both GPUs idle)
#   scripts/stage.sh stop      rank 0 drains and stops; rank 1 follows
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
recipe=$here/vendor/deepseek-v41-tensorfold-spark
config=${CONFIG:-$here/config/native.env}
log() { printf '[stage] %s\n' "$*"; }

PACK_REPO=${PACK_REPO:-LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
BASE_REPO=${BASE_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
# The 1017 pack ships no chat template; DeepSeek's encoding/encoding.py is byte-identical in the 1017 and the earlier
# V4.1-Flash release, so the template of Mia-AiLab's V4.1 pack applies unchanged.
TEMPLATE_REPO=${TEMPLATE_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}
PYIMAGE=${PYIMAGE:-dsv41-tensorfold:060}
PYSRC=/usr/local/lib/python3.12/dist-packages   # the reference source inside $PYIMAGE (TensorFold 0.6.0 + patches)

env_file() {
    [[ -f $config ]] || { log "no $config: run 'scripts/stage.sh setup' and edit it"; exit 1; }
    grep -qE '^[A-Z_]+=.*<' "$config" && { log "$config still has <placeholders>"; exit 1; }
    set -a; source "$config"; set +a
}
wssh() { ssh -o BatchMode=yes "$WORKER_SSH" "$@"; }
envs() { env | grep -E '^(TF_DSV41_|GLM53_TF_|TF_TP_|TF_COMM)' | grep -vE "^TF_DSV41_(ASSETS|${1:-NONE})=" | sed 's/^/-e /' | tr '\n' ' '; }

setup() {
    git -C "$here" submodule update --init --recursive
    [[ -f $config ]] || { cp "$here/config/native.env.example" "$config"; log "edit $config: every <placeholder>"; }
}

weights() {
    env_file
    hf download "$PACK_REPO" --local-dir "$MODEL_DIR" --max-workers 8
    hf download "$TEMPLATE_REPO" chat_template.jinja --local-dir "$MODEL_DIR"
    python3 - "$MODEL_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]; idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
miss = sorted({f for f in idx["weight_map"].values() if not os.path.exists(os.path.join(d, f))})
sys.exit(f"missing shards: {miss}") if miss else print(f"[stage] {len(set(idx['weight_map'].values()))} shards present")
PY
    wssh "mkdir -p '$MODEL_DIR'"
    rsync -a --info=progress2 --exclude .cache "$MODEL_DIR/" "$WORKER_SSH:$MODEL_DIR/"
}

engram() {
    env_file
    local src=${ENGRAM_SRC:-$(dirname "$ENGRAM_DIR")/dsv41-1017-engram-src} r1; r1=$(mktemp -d "$(dirname "$ENGRAM_DIR")/engram-r1.XXXX")
    mkdir -p "$src" "$ENGRAM_DIR"
    hf download "$BASE_REPO" model.safetensors.index.json config.json --local-dir "$src"
    # shellcheck disable=SC2046
    hf download "$BASE_REPO" $(python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --list) --local-dir "$src" --max-workers 4
    python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --config "$MODEL_DIR/config.json" --out "$ENGRAM_DIR" --rank 0 &
    python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --config "$MODEL_DIR/config.json" --out "$r1" --rank 1 &
    wait
    python3 -I "$recipe/scripts/pack_engram.py" --src "$src" --config "$MODEL_DIR/config.json" --check "$ENGRAM_DIR" --rank 0
    wssh "mkdir -p '$ENGRAM_DIR'"
    rsync -a --info=progress2 "$r1/" "$WORKER_SSH:$ENGRAM_DIR/" && rm -rf "$r1"
    log "done; $src (the base shards) can be deleted"
}

engine() {
    env_file
    local tree=$here/build/engine
    [[ -d $tree/.git ]] || "$here/engine/engine.sh" "$tree"
    docker build -f "$here/docker/Dockerfile" --build-arg ZIG_COMMIT="$(git -C "$tree" rev-parse --short HEAD)" -t "$IMAGE" "$tree"
    docker save "$IMAGE" | wssh docker load
}

pyimage() { # jayleaton's Python recipe image: only for the asset tools (no Python server runs)
    (cd "$recipe" && cp -n config/prod.env.example config/prod.env 2>/dev/null; IMAGE=$PYIMAGE docker build -f docker/Dockerfile -t "$PYIMAGE" .)
}

assets() {
    env_file
    local tools=/r/engine/zig/tools/zig py="docker run --rm -v $recipe:/r -v $MODEL_DIR:$MODEL_DIR:ro -v $ASSETS_DIR:/assets -w /r/engine/zig"
    mkdir -p "$ASSETS_DIR/aot"
    $py --entrypoint python "$PYIMAGE" $tools/triton_fill.py sigs --py $PYSRC --out /assets/sigs.json
    $py -e PYTHONPATH=$PYSRC -e TF_DSV41_CONTEXT="$TF_DSV41_CONTEXT" --entrypoint python "$PYIMAGE" $tools/dsv41_rope_tables.py \
        --config "$MODEL_DIR/config.json" --out /assets/rope.bin
    $py --entrypoint python "$PYIMAGE" $tools/dsv41_engram_host.py --py-src $PYSRC --config "$MODEL_DIR/config.json" \
        --tokenizer "$MODEL_DIR/tokenizer.json" --out /assets/engram-host.bin
    # every Triton variant the served env launches, at 4K prefill rows and at the 2K the engine falls back to when
    # memory at the worst serving point is short (a missing one stops a start with MissingTritonVariant)
    docker run --rm -v "$ASSETS_DIR:/assets" $(envs "PF_4K|PREFILL_CHUNK|PREFILL_ROWS") -e TF_DSV41_PF_4K=1 \
        -e TF_DSV41_PREFILL_CHUNK=4096 -e TF_DSV41_PREFILL_ROWS=4096 -e TF_DSV41_ASSETS=/assets -e TF_TP_WORLD=2 \
        "$IMAGE" tf-dsv41 aot-needs /assets/sigs.json /assets/needs-4k.json
    docker run --rm -v "$ASSETS_DIR:/assets" $(envs "PF_4K|PREFILL_CHUNK|PREFILL_ROWS") -e TF_DSV41_PF_4K=0 \
        -e TF_DSV41_PREFILL_CHUNK=2048 -e TF_DSV41_PREFILL_ROWS=2048 -e TF_DSV41_ASSETS=/assets -e TF_TP_WORLD=2 \
        "$IMAGE" tf-dsv41 aot-needs /assets/sigs.json /assets/needs-2k.json
    for n in needs-4k needs-2k; do
        $py --gpus all -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
            --entrypoint python "$PYIMAGE" $tools/triton_fill.py compile --py $PYSRC --needs /assets/$n.json \
            --options $tools/dsv41_triton_options.json --arch 121 --have /assets/aot --out /assets/aot
    done
    wssh "mkdir -p '$ASSETS_DIR'"
    rsync -a "$ASSETS_DIR/" "$WORKER_SSH:$ASSETS_DIR/"
}

cache() {
    env_file
    mkdir -p "$CACHE_DIR/dsv41-bias-vl"
    docker run --rm -v "$CACHE_DIR:/cache" -e HF_HUB_OFFLINE=0 --entrypoint python "$PYIMAGE" \
        -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch /cache/dsv41-bias-vl --repo "$BASE_REPO"
    cp "$recipe/config/pfdense-table.json" "$CACHE_DIR/"
    wssh "mkdir -p '$CACHE_DIR' '$STATE_DIR/sessions' '$PREPARED_DIR'"; mkdir -p "$STATE_DIR/sessions" "$PREPARED_DIR"
    rsync -a "$CACHE_DIR/" "$WORKER_SSH:$CACHE_DIR/"
}

start() {
    env_file
    local cfg; cfg=$(realpath "$config")
    wssh "mkdir -p '$(dirname "$cfg")'"; scp -q "$cfg" "$WORKER_SSH:$cfg"
    rsync -a --exclude build "$here/" "$WORKER_SSH:$here/"
    wssh "cd '$here/vendor/deepseek-v41-tensorfold-spark' && CONFIG='$cfg' nohup scripts/run-zig.sh 1 > /tmp/dsv41-r1.log 2>&1 < /dev/null &"
    sleep 2
    (cd "$recipe" && CONFIG="$cfg" nohup scripts/run-zig.sh 0 > /tmp/dsv41-r0.log 2>&1 < /dev/null &)
    log "starting; ready when /tmp/dsv41-r0.log says 'serving DeepSeek-V4.1-Flash-TF' (about 2 minutes, ~8 on a first start)"
}

stop() { env_file; docker stop -t $(( ${TF_DSV41_DRAIN_S:-20} + 60 )) tensorfold-zig-r0 >/dev/null; wssh "docker wait tensorfold-zig-r1 >/dev/null 2>&1 || true"; log stopped; }

case "${1:-}" in
setup) setup ;; weights) weights ;; engram) engram ;; engine) engine ;; pyimage) pyimage ;; assets) assets ;; cache) cache ;;
all) setup; weights; engram; engine; pyimage; assets; cache ;;
start) start ;; stop) stop ;;
*) sed -n '2,15p' "$0"; exit 2 ;;
esac
