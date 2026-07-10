#!/usr/bin/env python3
"""Generate real-model logit vectors with transformers (M2 gate).

Runs a handful of short prompts through google/gemma-4-26B-A4B-it (bf16,
local safetensors dir) and stores, per prompt, the token ids and the top-K
(id, logprob) of every position.  The C test compares the g4 engine output
against these.

NOTE: written for transformers >= 5.5 (Gemma4 support); untested until it
runs on the dev machine — expect to adjust the model-class import.

Output format (little-endian, tests/vectors/real_vectors.bin):
  u32 magic 'G4RV', u32 version=1, u32 n_prompts, u32 top_k
  per prompt: u32 n_tokens, i32 ids[n_tokens],
              then n_tokens records of top_k x { i32 id, f32 logprob }

Usage:
  python3 scripts/gen_logit_vectors.py --weights ~/g4/weights \
      [--out tests/vectors/real_vectors.bin] [--top-k 32] [--device cpu]
"""

import argparse
import struct

import torch


PROMPTS = [
    "The capital of France is",
    "Il Colosseo si trova a",
    "def fibonacci(n):\n",
    "2 + 2 * 3 =",
    "Il mare è blu perché",
    "In C, a segmentation fault happens when",
    "La quantizzazione a 2 bit di un modello MoE",
    "Q: What is the boiling point of water in Celsius?\nA:",
]


def load_model(path, device):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = None
    errors = []
    for cls_name in ("Gemma4ForConditionalGeneration",
                     "AutoModelForCausalLM", "AutoModel"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            kwargs = dict(dtype=torch.bfloat16)
            if device == "auto":
                kwargs["device_map"] = "auto"
            model = cls.from_pretrained(path, **kwargs)
            if device not in ("auto",):
                model = model.to(device)
            print(f"loaded with {cls_name}")
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {e}")
    if model is None:
        raise SystemExit("cannot load model:\n" + "\n".join(errors))
    model.eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="tests/vectors/real_vectors.bin")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda | auto (accelerate offload)")
    args = ap.parse_args()

    tok, model = load_model(args.weights, args.device)

    out = bytearray()
    out += b"G4RV" + struct.pack("<III", 1, len(PROMPTS), args.top_k)
    for text in PROMPTS:
        ids = tok(text, return_tensors="pt").input_ids  # includes BOS
        with torch.no_grad():
            result = model(input_ids=ids.to(model.device))
        logits = result.logits[0].float()             # [T, vocab]
        logprobs = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logprobs, k=args.top_k, dim=-1)
        seq = ids[0].tolist()
        print(f"{text!r}: {len(seq)} tokens, "
              f"greedy next = {top.indices[-1, 0].item()} "
              f"({tok.decode([top.indices[-1, 0].item()])!r})")
        out += struct.pack("<I", len(seq))
        out += struct.pack(f"<{len(seq)}i", *seq)
        for t in range(len(seq)):
            for k in range(args.top_k):
                out += struct.pack("<if", int(top.indices[t, k]),
                                   float(top.values[t, k]))
    with open(args.out, "wb") as fp:
        fp.write(out)
    print(f"wrote {args.out}: {len(out)} bytes")


if __name__ == "__main__":
    main()
