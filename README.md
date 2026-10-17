# DeepSeek-V4.1-Flash-1017 on two DGX Sparks

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT), served at
TP=2 on two NVIDIA DGX Sparks (GB10, 128 GB each) over their CX7 link, behind an OpenAI-compatible API.

This is a thin recipe on top of [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)
(Apache-2.0). That recipe serves the earlier V4.1-Flash on TensorFold 0.6.0 with its own patches. It is included
unmodified as the submodule `vendor/deepseek-v41-tensorfold-spark`. This repository adds what the 1017 release
needs, and one fix for GB10:

- **The 1017 pack.** [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  is EXL3 at 2.9 bpw, 196 GiB. Its tensors match Mia-AiLab's V4.1 pack, which the recipe loads. The 1017 release
  has the same architecture as V4.1-Flash, with new weights.
- **The 1017 Engram tables**, from shards 47 and 48 of the base repo, which changed in this release. They are packed
  per rank with the recipe's `pack_engram.py`.
- **The 1017 image routing bias**, fetched from the 1017 repo. The recipe would otherwise fetch the older release's.
- **A chat template.** The 1017 pack ships none, and the server stops without one. DeepSeek's `encoding/encoding.py`
  is byte-identical in the two releases, so the template of Mia-AiLab's pack applies unchanged.
- **`patches/0001-serve-nccl-cumem-off.patch`** sets `NCCL_CUMEM_HOST_ENABLE=0` and `NCCL_CUMEM_ENABLE=0` in both
  ranks' containers. On GB10, the first boot left the worker's driver with `NV_ERR_NO_MEMORY` from
  `_memdescAllocInternal`, and every later NCCL start then failed with "unhandled system error" until a reboot.
  With the settings off, starts come up clean. The same fix holds for GLM-5.3-Flash on the same pair.

## Measured

Recipe harness ([recipe-harness](https://github.com/BobClawblaw/recipe-harness), target `harness/dsv41-1017-gb10.toml`,
plan `harness/dsv41-speed.toml`) on two DGX Sparks, recipe defaults (4 slots, DSpark drafts, replay prefill).
Aggregate tok/s for prose / code / structured / list, then first-text latency:

| Streams | tok/s | First text |
|---|---|---|
| 1 | 54.7 / 100.6 / 62.6 / 71.3 | 0.19-0.26 s |
| 2 | 77.4 / 126.5 / 84.0 / 142.6 | 0.30-0.39 s |
| 4 | 103.5 / 155.8 / 109.0 / 173.2 | 0.41-0.64 s |

- **Memory:** about 8 GiB available on the head and 10 GiB on the worker while serving, with 4 slots at a
  300,000-token window.
- **Leaderboard:** the TensorFold community benchmark (attached, so unranked) measured chat at 63.6 tok/s (T=1) and
  63.7 tok/s (T=0), first text 0.20 s. Its `code` workload sends a bare prompt to `/v1/completions`, and DeepSeek-V4.1's
  greedy first token there is end-of-sequence, so `code` is not measured.
- **Quality:** see [Quality](#quality).

## Requirements

As the recipe's [docs/INSTALL.md](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/main/docs/INSTALL.md):
two DGX Sparks with their CX7 ports cabled and addressed, Docker with the NVIDIA runtime, and passwordless ssh from
the head to the worker over the link.

- **Each node's NVMe:** about 211 GB for the pack, 95 GB for its Engram half, about 105 GB for the prepared folders
  (written on first start), and the session tier (128 GiB by default).
- **The head, while staging:** 190 GB more for the base Engram shards; they can be deleted after packing.
- **The GPUs:** nothing else on them.

## Run

```bash
git clone --recurse-submodules https://github.com/BobClawblaw/deepseek-v41-flash-1017-2x-dgx-sparks.git
cd deepseek-v41-flash-1017-2x-dgx-sparks
scripts/stage.sh setup        # applies the patch, writes vendor/.../config/prod.env from config/prod.env.example
$EDITOR vendor/deepseek-v41-tensorfold-spark/config/prod.env   # WORKER_SSH, HEAD_IP, paths, NIC / HCA names
scripts/stage.sh weights      # pack + chat template, then onto the worker (~211 GB)
scripts/stage.sh engram       # 1017 Engram shards (~190 GB), each rank's half, rank 1's onto the worker
scripts/stage.sh build        # the recipe's image on both nodes
scripts/stage.sh bias         # the 1017 image routing bias into both nodes' cache volume
cd vendor/deepseek-v41-tensorfold-spark
scripts/serve.sh prebuild && scripts/serve.sh preflight && scripts/serve.sh start   # GPUs idle on both nodes
```

The first start writes the prepared folders (about 6 minutes); later starts take about a minute. Check the answer to a
known question before you trust the server:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "temperature": 0, "reasoning_effort": "low"}'
```

`scripts/serve.sh status | logs [0|1] | stop | restart` as in the recipe. Thinking is on by default (effort 75);
`"chat_template_kwargs": {"enable_thinking": false}` turns it off for a request.

## Quality

Pending: the harness's quality plan (`harness/dsv41-quality.toml`) runs with thinking off and on. The results go here.

## Known issues

- **The recipe's own list** ([docs/KNOWN-ISSUES.md](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/main/docs/KNOWN-ISSUES.md))
  applies. In particular, prefill is the recipe's approximate `replay` by default; `TF_DSV41_PREFILL=full` is exact
  and slower.
- **Memory headroom is small by design.** The recipe plans for a 4-5 GiB floor per node.
- **The 1017 pack is new.** It was published the day this recipe was made, so it has had less use than the V4.1 packs.

## License

Apache-2.0 (see LICENSE and NOTICE). The model weights keep their own licenses (MIT for DeepSeek's release).
