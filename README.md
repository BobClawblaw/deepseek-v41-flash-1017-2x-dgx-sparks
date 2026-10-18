# DeepSeek-V4.1-Flash-1017 on two DGX Sparks, native TensorFold

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT) on the
native TensorFold engine (Zig and CUDA, built from TensorFold's main branch). It runs at TP=2 on two NVIDIA DGX Sparks
(GB10, 128 GB each) over their CX7 link, with text, **images** (the model's vision tower) and **video** (sampled frames).
It serves an OpenAI-compatible API.

## What this is

TensorFold's main branch has no DeepSeek-V4.1 family yet. This recipe builds one from public work and adds what the
1017 release and video need:

- **The engine** (`engine/engine.sh`) assembles its source from four parts:
  1. TensorFold main at `9ccf4727`.
  2. jayleaton's DeepSeek-V4.1 series ([ashhart/TensorFold #727](https://github.com/ashhart/TensorFold/pull/727) and the
     PRs it stacks on), merged.
  3. The series' newer engine with the vision tower. This is the "kit", engine `9131e10a`, from
     [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark), the submodule.
     The script overlays the files that `engine/kit-overlay.txt` lists.
  4. This repository's patches (`engine/patches/`):
     - the kit on main's runtime, lane host and server;
     - native images on main's server;
     - the 1017 pack's fp16 tensors;
     - video.
- **The 1017 pack** is [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw):
  EXL3 at 2.9 bpw, 196 GiB, in Mia-AiLab's layout. Two of its tensors are stored as fp16 where the engine expected
  fp32: the router bias and DSpark's confidence head. The patches widen them exactly at load.
- **The 1017 Engram tables**, from shards 47 and 48 of the base repo. They changed in this release. Each rank keeps its
  half on its own NVMe.
- **The 1017 image routing bias**, fetched from the 1017 repo.
- **A chat template.** The pack ships none. DeepSeek's `encoding/encoding.py` is byte-identical in the 1017 and the
  earlier release, so Mia-AiLab's template applies.
- **Video.** DeepSeek-V4.1 has no video input of its own. A `video_url` part is decoded by TensorFold's `libtfvideo`
  (FFmpeg) and sampled evenly at up to 2 frames a second, up to `TF_DSV41_VIDEO_MAX_FRAMES` (8). Each frame goes in as
  an image, after a "*t* seconds" line.

## Measured

Recipe harness ([recipe-harness](https://github.com/BobClawblaw/recipe-harness)), target
`harness/dsv41-1017-gb10-native-vision8.toml`, plan `harness/dsv41-speed8.toml`, on two DGX Sparks. Settings: 8 slots,
DSpark drafts, native images. Aggregate tok/s for prose / code / structured / list, then first-text latency:

| Streams | tok/s | First text |
|---|---|---|
| 1 | 55.4 / 103.5 / 62.7 / 70.9 | 0.19-0.26 s |
| 2 | 81.5 / 126.2 / 89.4 / 142.7 | 0.30-0.48 s |
| 4 | 108.8 / 163.8 / 113.7 / 180.4 | 0.71-0.85 s |
| 8 | 134.8 / 202.7 / 143.6 / 203.3 | 1.35-1.84 s |

8 slots with a 400K pool and a 2 GiB prefill-dip reserve (`config/native.env.example`, harness plan `harness/dsv41-speed8.toml`). With 4 slots and the kit's 3 GiB reserve the pool had to stay at 200K, and 4 streams measured 106.5 / 161.6 / 111.9 / 178.7.

The same build without the vision tower (the series alone) measured 98.1 / 158.5 tok/s on code at 1 / 4 streams.

Checks run on the served model:
- 17 x 23 = 391.
- DeepSeek's own example image is described correctly ("five fresh, bright orange carrots with green stems").
- A 4-second clip of four solid colours came through as 8 frames at 0.2 to 3.9 seconds. Every colour was named at
  its time.

Quality: see [Quality](#quality).

## Requirements

- **Hardware:** two DGX Sparks with their CX7 ports cabled and addressed, Docker with the NVIDIA runtime and RDMA
  devices, and passwordless ssh from the head to the worker over the link.
- **Each node's NVMe:**

  | What | Size |
  |---|---|
  | Pack | ~211 GB |
  | Engram half | ~95 GB |
  | Prepared weights (written on the first start) | ~100 GB |
  | Assets | ~1 GB |
  | Session tier | 128 GiB |

  The head needs ~190 GB more while staging, for the base Engram shards.
- **Nothing else on the GPUs.**

## Run

```bash
git clone --recurse-submodules https://github.com/BobClawblaw/deepseek-v41-flash-1017-2x-dgx-sparks.git
cd deepseek-v41-flash-1017-2x-dgx-sparks
scripts/stage.sh setup      # writes config/native.env: fill in every <placeholder>
scripts/stage.sh all        # weights, Engram, engine image, asset tools image, assets, cache (hours, mostly downloads)
scripts/stage.sh start      # rank 1 on the worker, rank 0 here
```

- **When it's ready:** rank 0's log (`/tmp/dsv41-r0.log`) prints `serving DeepSeek-V4.1-Flash-TF at
  http://localhost:8000/v1`. That takes about 8 minutes on a first start, while it writes the prepared weights, and
  about 2 minutes after that.
- **Stopping:** `scripts/stage.sh stop` lets rank 0 drain, then rank 1 follows.

Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "temperature": 0, "reasoning_effort": "low"}'
```

Images and video are OpenAI content parts: `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`
and `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`. They also accept public `https://`
URLs. Thinking is on by default; `"chat_template_kwargs": {"enable_thinking": false}` turns it off for a request.

## Quality

Pending. The harness quality plan (`harness/dsv41-quality.toml`, thinking off and on) is run on this build, and the
results go here.

## Known issues

- **A small KV pool.** With the vision tower loaded, rank 0 is the tight rank. At the kit's 3 GiB prefill-dip reserve,
  the boot gate passed only 200K pool positions for 4 slots, and that was within 0.01 GiB of its floor. This config
  lowers the reserve to 2 GiB (`TF_DSV41_SERVE_DIP_GIB`) for 8 slots and a 400K pool, which leaves ~0.7 GiB of margin.
  Long prompts still run in 2,048-row prefill segments at the tightest point. The gate prints the largest pool that
  fits when it refuses one.
- **The worker's driver after a failed start.** On our pair, every start that failed (a memory-gate refusal, a missing
  kernel variant) left the worker's NVIDIA driver logging `NV_ERR_NO_MEMORY`. The next start then failed in NCCL's
  memory registration (`ibv_reg_mr ... Cannot allocate memory`) until the worker was rebooted. Reboot it before
  retrying a failed start.
- **Missing kernel variants.** Changing a serving setting can need Triton variants the AOT set lacks. A start then
  stops with `MissingTritonVariant`. `scripts/stage.sh assets` enumerates and compiles the set again; variants already
  compiled are kept.
- **Structured output** in this engine also needs `libtfgrammar.so` beside the server (TensorFold's
  `zig build grammar -Dxgrammar=...`). Without it, requests with a response format are refused.
- **The Dockerfile route.** The served build was made with the same sources and steps, but outside Docker.
  `docker/Dockerfile` has not been run end to end yet. Report what breaks.

## License

Apache-2.0 (see LICENSE and NOTICE). The model weights keep their own licenses: MIT for DeepSeek's release, MIT for
the EXL3 packs.
