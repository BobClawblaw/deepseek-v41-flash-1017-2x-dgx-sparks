# The native DeepSeek-V4.1 engine for two DGX Sparks (sm_121, aarch64), built from engine/engine.sh's tree:
#   engine/engine.sh build/engine && docker build -f docker/Dockerfile -t dsv41-1017-native build/engine
# The build stage is jayleaton/deepseek-v41-tensorfold-spark's docker/zig/Dockerfile (Apache-2.0) with the binaries
# of this tree (`tf-dsv41` carries follow and aot-needs) and libtfvideo for video input; the runtime stage is its
# runtime with libtfvideo on the loader path and `tf-dsv41-m1` (the launcher's name) linked to `tf-dsv41`.
# Build context: src/ is an exported CUDA DeepSeek Zig port; fatbins/ holds optional sm_121 kernels.
# See docs/ZIG-SERVE.md for source availability and required runtime assets.
# must match the nvcc the gated fatbins were built with when fatbins/ is empty (13.x)
ARG CUDA_DEVEL=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG CUDA_RUNTIME=nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04

FROM ${CUDA_DEVEL} AS build
ARG ZIG_VERSION=0.17.0
ARG ZIG_URL=
ARG ZIG_TARGET=aarch64-linux-gnu
# the GB10's ARMv8 crypto extensions for the host code: std's SHA-256 (prepared files, PREPARED_VERIFY) runs in software
# on the generic aarch64 baseline (adds instructions only; no float code changes)
ARG ZIG_CPU=baseline+sha2+aes
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; url="${ZIG_URL:-https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz}"; \
    mkdir -p /opt/zig && curl -fsSL "$url" | tar -xJ -C /opt/zig --strip-components=1; \
    /opt/zig/zig version
ENV PATH=/opt/zig:/usr/local/cuda/bin:$PATH
COPY . /ctx/src
WORKDIR /ctx/src
RUN set -eu; \
    nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 \
        -gencode=arch=compute_90,code=compute_90 -o /tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu; \
    zig build -Dtarget="$ZIG_TARGET" -Dcpu="$ZIG_CPU" -Dnvcc=/usr/local/cuda/bin/nvcc -Dtp-fatbin=/tp_mailbox.fatbin \
        -Dsm=121 -Doptimize=ReleaseSafe -j4 --prefix /out install; \
    test -x /out/bin/tensorfold-dsv41 && test -x /out/bin/tf-dsv41
# libtfvideo (video input): main's pinned static FFmpeg, then main's `zig build video`
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config perl && rm -rf /var/lib/apt/lists/*
RUN set -eu; tools/zig/build_ffmpeg.sh /opt/ffmpeg /tmp/ffmpeg-src; \
    zig build -Dtarget="$ZIG_TARGET" -Dcpu="$ZIG_CPU" -Doptimize=ReleaseSafe -Dffmpeg=/opt/ffmpeg --prefix /out video; \
    find /out -name libtfvideo.so | head -1 | xargs -I{} cp {} /libtfvideo.so; test -f /libtfvideo.so

FROM ${CUDA_RUNTIME}
ARG ZIG_COMMIT=unknown
# libnccl2 from the CUDA apt repo the nvidia images ship configured; libibverbs + the mlx5 provider for RoCE; cuBLAS /
# cuBLASLt pinned at 13.6.0.2 (libcublas-13-3, the same repo): the build the vision tower's bit-exact proof ran on, dlopened
# only with TF_DSV41_IMAGES=native. Its directory goes on the loader path (the base image's points at its own CUDA).
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers rdma-core \
        curl ca-certificates libcublas-13-3=13.6.0.2-1 \
    && apt-mark hold libcublas-13-3 \
    && dirname "$(dpkg -L libcublas-13-3 | grep '/libcublas\.so\.13$')" > /etc/ld.so.conf.d/990_cublas-13-3.conf && ldconfig \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/tensorfold-dsv41 /out/bin/tf-dsv41 /opt/tensorfold/bin/
RUN ln -s tf-dsv41 /opt/tensorfold/bin/tf-dsv41-m1
COPY --from=build /tp_mailbox.fatbin /opt/tensorfold/share/tp_mailbox.fatbin
COPY --from=build /libtfvideo.so /opt/tensorfold/lib/libtfvideo.so
RUN echo /opt/tensorfold/lib > /etc/ld.so.conf.d/980_tensorfold.conf && ldconfig
# libproof: every library name the engine can dlopen is covered, each resolves here with its whole NEEDED chain, and
# cuBLAS / cuBLASLt are the pinned 13.6.0.2 files by sha256; any miss or other build fails the image build
COPY <<'EOF' /opt/tensorfold/share/libproof.sh
#!/bin/bash
set -u
need="libnccl.so.2 libibverbs.so.1 libcublas.so.13 libcublasLt.so.13"
pins="8062c27c479c249292b48c4bc73af1deb22edee4dfff191fb48a7491ac506410  libcublas.so.13.6.0.2
4198b8726952fbf25a4e78f07178d66509aca4919b7324e935fe3915bff3efc8  libcublasLt.so.13.6.0.2"
known=" libtfgrammar.so libtfvideo.so libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 libcuda.so.1 libcublas.so libcublasLt.so $need "
fail=0
names=$(grep -aohE 'lib[A-Za-z0-9_]+\.so(\.[0-9]+)*' /opt/tensorfold/bin/* | sort -u | tr '\n' ' ')
echo "libproof: engine dlopen names: $names"
for n in $names; do case "$known" in *" $n "*) ;; *) echo "libproof: UNCOVERED dlopen name $n"; fail=1 ;; esac; done
for s in $need; do
    f=$(ldconfig -p | awk -v s="$s" '$1 == s {print $NF; exit}')
    [ -n "$f" ] || { echo "libproof: $s MISSING"; fail=1; continue; }
    r=$(readlink -f "$f"); h=$(sha256sum "$r" | cut -c1-64); u=$(ldd "$r" 2>&1 | grep -c 'not found'); note=""
    [ "$u" = 0 ] || { note=" NEEDED NOT RESOLVED"; fail=1; }
    case $s in libcublas*) if printf '%s\n' "$pins" | grep -qx "$h  ${r##*/}"; then note="$note = pin"; else note="$note != PIN"; fail=1; fi ;; esac
    echo "libproof: $s -> $r ${h:0:12}$note"
done
ls /usr/lib/*-linux-gnu/libibverbs/ 2>/dev/null | grep -q mlx5 || { echo "libproof: mlx5 provider MISSING"; fail=1; }
[ "$fail" = 0 ] && echo "LIBPROOF PASS" || echo "LIBPROOF FAIL"
exit "$fail"
EOF
RUN bash /opt/tensorfold/share/libproof.sh
ENV PATH=/opt/tensorfold/bin:$PATH \
    TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin
LABEL org.tensorfold.zig-commit="${ZIG_COMMIT}"
