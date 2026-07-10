# g4 — motore di inferenza per Gemma 4 26B-A4B

Motore minimale in C11 per Gemma 4 26B-A4B con quantizzazione asimmetrica
degli esperti routed e disk KV cache, sul modello concettuale di ds4.

I documenti di progetto sono in [`../gemma4-port/`](../gemma4-port/):

1. `01-confronto-architetturale.md` — DeepSeek V4 Flash vs Gemma 4 26B-A4B
2. `02-schema-quantizzazione.md` — ricetta IQ2_XXS/Q2_K/Q8_0 (~8.6 GiB)
3. `03-disk-kv-cache.md` — formato G4KV/G4SP e politiche di frontiera
4. `04-piano-implementazione.md` — milestone M0-M7

## Stato: M1 + tokenizer M3 validati

| Modulo | Contenuto | Stato |
|---|---|---|
| `g4_quants.[ch]` | quantizzatori IQ2_XXS/Q2_K/Q4_K/Q8_0 (da `gguf-tools/quants.[ch]`, simboli rinominati) | fatto |
| `g4_dequant.c` | dequantizzazione di riferimento, convenzioni di decode identiche ai kernel ds4.c | fatto |
| `g4_gguf.[ch]` | lettore/scrittore GGUF v3 minimale (chiavi `gemma4.*`) | fatto |
| `g4_kvstore.[ch]` | header G4KV 48 byte, SHA1, eviction score | fatto (contenitore; store completo a M6) |
| `tests/g4_test.c` | round-trip quant (incluso il caso padding down 704→768), round-trip GGUF, header/eviction KVG | verde |
| `g4.[ch]` | loader del modello dal GGUF + forward pass f32 di riferimento (attn locale/globale K=V, RoPE parziale, MoE top-8 rinormalizzato, ramo denso, GeGLU, softcap) | fatto |
| `tests/g4_toy_test.c` + `tests/vectors/toy.gguf` | **M1**: il forward C riproduce i logits della reference JAX vera (modello giocattolo, 6 layer LLLLLG): max diff 5e-6, argmax 21/21 | verde |
| `g4_tokenizer.[ch]` | BPE SentencePiece-style: normalizer ▁, merges per rank, byte-fallback, special tokens greedy | fatto |
| `tests/g4_tok_test.c` + `tests/vectors/tok_vectors.bin` | **M3 (encode)**: 30/30 casi identici a HF `tokenizers` su tokenizer.json reale (multilingua, emoji, byte-fallback, specials); tabella rigenerabile con `scripts/gen_tokenizer_vectors.py` | verde |

```sh
make test
```

Il GGUF giocattolo è generato dai moduli JAX veri del fork gemma:

```sh
cd ../../gemma && python3 g4ref/gen_toy_vectors.py ../ds4/gemma4/tests/vectors/toy.gguf
```

Prossima milestone (doc 04 §4): M2 — convertitore `g4-quantize` dai
safetensors reali e vettori logits via transformers (richiede i pesi da
48 GiB sulla macchina di sviluppo).
