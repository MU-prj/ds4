#!/usr/bin/env python3
"""Build a toy-scale synthetic HF checkpoint from tests/vectors/toy.gguf.

Emits config.json + model.safetensors(.index.json) with the exact tensor
names of google/gemma-4-26B-A4B-it (model.language_model.*) and the layout
conventions g4-quantize expects, but with the toy dims and the toy weights.
Converting it back with g4-quantize must therefore produce a GGUF on which
the engine reproduces the JAX reference logits (tests/g4_toy_test).

Layout conventions encoded here (and auto-verified by shape in the
converter, since d_model differs from every other dim):
  - 2D `.weight` tensors: torch nn.Linear [out_features, in_features],
    row-major == GGUF rows with dims[0] = in_features (no transpose);
  - experts.gate_up_proj: [E, D, 2*Hexp], gate = first Hexp columns;
  - experts.down_proj:    [E, Hexp, D].

Usage: python3 scripts/gen_synthetic_hf.py OUT_DIR [--f32]
"""

import json
import os
import struct
import sys

import numpy as np

TOY = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors",
                   "toy.gguf")


def read_gguf(path):
    """Minimal GGUF v3 reader: returns (meta, {name: np.array})."""
    f = open(path, "rb")
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", f.read(24))
    assert magic == 0x46554747 and ver == 3

    def rd_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()

    scal = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
            10: "<Q", 11: "<q", 12: "<d"}
    meta = {}
    for _ in range(n_kv):
        k = rd_str()
        (t,) = struct.unpack("<I", f.read(4))
        if t in scal:
            (v,) = struct.unpack(scal[t], f.read(struct.calcsize(scal[t])))
        elif t == 7:
            v = f.read(1) != b"\0"
        elif t == 8:
            v = rd_str()
        elif t == 9:
            (et, cnt) = struct.unpack("<IQ", f.read(12))
            if et == 8:
                v = [rd_str() for _ in range(cnt)]
            else:
                sz = struct.calcsize(scal[et])
                v = list(struct.unpack(f"<{cnt}{scal[et][1]}", f.read(sz * cnt)))
        meta[k] = v
    infos = []
    for _ in range(n_t):
        name = rd_str()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        t, off = struct.unpack("<IQ", f.read(12))
        infos.append((name, dims, t, off))
    data0 = (f.tell() + 31) // 32 * 32
    tensors = {}
    for name, dims, t, off in infos:
        n = 1
        for d in dims:
            n *= d
        f.seek(data0 + off)
        if t == 0:
            arr = np.frombuffer(f.read(4 * n), dtype=np.float32)
        elif t == 26:
            arr = np.frombuffer(f.read(4 * n), dtype=np.int32)
        else:
            raise SystemExit(f"unexpected tensor type {t} in toy gguf")
        # GGUF dims[0] is the contiguous axis -> numpy shape is reversed.
        tensors[name] = arr.reshape(tuple(reversed(dims)))
    return meta, tensors


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: gen_synthetic_hf.py OUT_DIR [--f32]")
    out = sys.argv[1]
    use_f32 = "--f32" in sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    meta, g = read_gguf(TOY)

    L = meta["gemma4.block_count"]
    D = meta["gemma4.embedding_length"]
    hf = {}

    hf["model.language_model.embed_tokens.weight"] = g["token_embd.weight"]
    hf["model.language_model.norm.weight"] = g["output_norm.weight"]
    for i in range(L):
        p = f"blk.{i}."
        o = f"model.language_model.layers.{i}."
        hf[o + "input_layernorm.weight"] = g[p + "attn_norm.weight"]
        hf[o + "post_attention_layernorm.weight"] = g[p + "post_attn_norm.weight"]
        hf[o + "layer_scalar"] = g[p + "skip_scale"]
        hf[o + "self_attn.q_proj.weight"] = g[p + "attn_q.weight"]
        hf[o + "self_attn.k_proj.weight"] = g[p + "attn_k.weight"]
        if p + "attn_v.weight" in g:
            hf[o + "self_attn.v_proj.weight"] = g[p + "attn_v.weight"]
        hf[o + "self_attn.o_proj.weight"] = g[p + "attn_output.weight"]
        hf[o + "self_attn.q_norm.weight"] = g[p + "attn_q_norm.weight"]
        hf[o + "self_attn.k_norm.weight"] = g[p + "attn_k_norm.weight"]
        hf[o + "router.proj.weight"] = g[p + "ffn_gate_inp.weight"]
        hf[o + "router.scale"] = g[p + "router_scale"]
        hf[o + "router.per_expert_scale"] = g[p + "per_expert_scale"]
        # gate/up: gguf [E, Hexp, D] (numpy) -> hf [E, D, 2*Hexp]
        gate = g[p + "ffn_gate_exps.weight"]
        up = g[p + "ffn_up_exps.weight"]
        hf[o + "experts.gate_up_proj"] = np.concatenate(
            [np.transpose(gate, (0, 2, 1)), np.transpose(up, (0, 2, 1))],
            axis=2)
        # down: gguf [E, D, Hexp] -> hf [E, Hexp, D]
        hf[o + "experts.down_proj"] = np.transpose(
            g[p + "ffn_down_exps.weight"], (0, 2, 1))
        hf[o + "mlp.gate_proj.weight"] = g[p + "ffn_gate_shexp.weight"]
        hf[o + "mlp.up_proj.weight"] = g[p + "ffn_up_shexp.weight"]
        hf[o + "mlp.down_proj.weight"] = g[p + "ffn_down_shexp.weight"]
        hf[o + "pre_feedforward_layernorm.weight"] = g[p + "ffn_norm.weight"]
        hf[o + "pre_feedforward_layernorm_2.weight"] = g[p + "ffn_norm_shexp.weight"]
        hf[o + "post_feedforward_layernorm_1.weight"] = g[p + "post_ffn_norm_moe.weight"]
        hf[o + "post_feedforward_layernorm_2.weight"] = g[p + "post_ffn_norm_shexp.weight"]
        hf[o + "post_feedforward_layernorm.weight"] = g[p + "post_ffn_norm.weight"]
    # A vision tensor the converter must skip.
    hf["model.vision_tower.std_scale"] = np.zeros(4, dtype=np.float32)

    # safetensors single shard
    if use_f32:
        dtype_name, to_bytes = "F32", lambda a: a.astype(np.float32).tobytes()
    else:
        def to_bf16(a):
            u = a.astype(np.float32).view(np.uint32)
            r = ((u >> 16) + ((u >> 15) & 1)).astype(np.uint32)  # rne approx
            return (r & 0xFFFF).astype(np.uint16).tobytes()
        dtype_name, to_bytes = "BF16", to_bf16

    header = {}
    blobs = []
    off = 0
    for name, arr in hf.items():
        raw = to_bytes(arr)
        header[name] = {"dtype": dtype_name, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    pad = (-len(hj)) % 8
    hj += b" " * pad
    with open(os.path.join(out, "model.safetensors"), "wb") as fp:
        fp.write(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))

    json.dump({"metadata": {"total_size": off},
               "weight_map": {n: "model.safetensors" for n in hf}},
              open(os.path.join(out, "model.safetensors.index.json"), "w"))

    cfg = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "attention_k_eq_v": bool(meta["gemma4.attention.k_eq_v_global"]),
            "final_logit_softcapping": meta["gemma4.final_logit_softcap"],
            "global_head_dim": meta["gemma4.attention.global_key_length"],
            "head_dim": meta["gemma4.attention.key_length"],
            "hidden_size": D,
            "intermediate_size": meta["gemma4.dense_ffn_length"],
            "max_position_embeddings": meta["gemma4.context_length"],
            "moe_intermediate_size": meta["gemma4.expert_ffn_length"],
            "num_attention_heads": meta["gemma4.attention.head_count"],
            "num_experts": meta["gemma4.expert_count"],
            "num_global_key_value_heads": meta["gemma4.attention.global_head_count_kv"],
            "num_hidden_layers": L,
            "num_key_value_heads": meta["gemma4.attention.head_count_kv"],
            "rms_norm_eps": meta["gemma4.attention.layer_norm_rms_epsilon"],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": meta["gemma4.rope.global.partial_factor"],
                    "rope_theta": meta["gemma4.rope.global.freq_base"],
                },
                "sliding_attention": {
                    "rope_theta": meta["gemma4.rope.local.freq_base"],
                },
            },
            "sliding_window": meta["gemma4.attention.sliding_window"],
            "top_k_experts": meta["gemma4.expert_used_count"],
            "vocab_size": meta["gemma4.vocab_size"],
        },
        "vision_config": {"num_hidden_layers": 27},
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"wrote synthetic HF checkpoint ({dtype_name}) to {out}: "
          f"{len(hf)} tensors, {off} data bytes")


if __name__ == "__main__":
    main()
