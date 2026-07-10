#!/usr/bin/env python3
"""Build the g4 tokenizer table blob + encode test vectors.

Reads the HF tokenizer.json committed in gemma4-port/hf-metadata/ and writes:
  tests/vectors/tok_table.bin   (vocab + merges + added tokens; ~9 MB,
                                 regenerated on demand, not committed)
  tests/vectors/tok_vectors.bin (encode test cases, committed)

Table blob format (little-endian):
  magic "G4TK", u32 version=1, u32 vocab_size, u32 n_merges, u32 n_added
  vocab_size x { u16 len, bytes }              token id = record index
  n_merges  x { u32 left, u32 right, u32 result }
  n_added   x { u32 id }                        (ids of added/special tokens)

Vector file: u32 n; n x { u32 text_len, bytes, u32 n_ids, u32 ids[] }.
"""

import json
import struct
import sys

from tokenizers import Tokenizer

META = "../gemma4-port/hf-metadata/tokenizer.json"


def main():
    tok = Tokenizer.from_file(META)
    spec = json.load(open(META))
    model = spec["model"]
    vocab = model["vocab"]  # dict str -> id
    merges = model["merges"]
    added = spec.get("added_tokens", [])

    n_vocab = max(vocab.values()) + 1
    inv = [""] * n_vocab
    for s, i in vocab.items():
        inv[i] = s

    blob = bytearray()
    blob += b"G4TK" + struct.pack("<IIII", 1, n_vocab, len(merges), len(added))
    for s in inv:
        raw = s.encode("utf-8")
        assert len(raw) < 65536
        blob += struct.pack("<H", len(raw)) + raw
    skipped = 0
    for m in merges:
        left, right = (m if isinstance(m, list) else m.split(" ", 1))
        li, ri = vocab.get(left), vocab.get(right)
        res = vocab.get(left + right)
        if li is None or ri is None or res is None:
            skipped += 1
            continue
        blob += struct.pack("<III", li, ri, res)
    if skipped:
        # patch n_merges in the header
        struct.pack_into("<I", blob, 12, len(merges) - skipped)
        print(f"warning: skipped {skipped} merges with out-of-vocab parts")
    for a in added:
        blob += struct.pack("<I", a["id"])
    open("tests/vectors/tok_table.bin", "wb").write(blob)

    cases = [
        "", " ", "  ", "a", "Hello world", "Hello, world!",
        "ciao, come stai?", "L'apostrofo dell'italiano è qui",
        "  due  spazi   e tre", "\ttab\te\ta capo\n", "\n\n\n",
        "def f(x):\n    return x * 2  # commento\n",
        "x=1;\ny=2;\nprintf(\"%d\\n\", x+y);\n",
        "日本語のテストです", "中文分词测试", "Ελληνικά και русский",
        "🤖 emoji 🚀🔥 test 👍🏽", "café naïve résumé",
        "<|turn>user\nCiao<turn|>\n<|turn>model\n",
        "<|tool_call>call:get_weather{\"city\": \"Roma\"}<tool_call|>",
        "<|channel>thought\npensiero<channel|>risposta<turn|>",
        "1234567890 3.14159 -42", "CamelCaseWord snake_case_word",
        "una frase più lunga che serve a esercitare merge multipli del BPE "
        "con parole comuni e meno comuni come pneumotorace e ossimoro",
        "mixed English e italiano insieme with code `let x = 5;`",
        "<bos><eos><pad><unk>", "http://example.com/path?q=1&r=2",
        "▁già con markup ▁ strano",
        "\x00byte nullo?", "ⅷ Ⅸ unicode numerali",
    ]

    out = bytearray()
    out += struct.pack("<I", len(cases))
    for c in cases:
        ids = tok.encode(c, add_special_tokens=False).ids
        raw = c.encode("utf-8")
        out += struct.pack("<I", len(raw)) + raw
        out += struct.pack("<I", len(ids))
        out += struct.pack(f"<{len(ids)}I", *ids)
    open("tests/vectors/tok_vectors.bin", "wb").write(out)
    print(f"table: {len(blob)} bytes, vectors: {len(out)} bytes, "
          f"{len(cases)} cases")


if __name__ == "__main__":
    sys.exit(main())
