# g4 — motore di inferenza per Gemma 4 26B-A4B

Motore minimale in C11 per Gemma 4 26B-A4B con quantizzazione asimmetrica
degli esperti routed e disk KV cache, sul modello concettuale di ds4.

I documenti di progetto sono in [`../gemma4-port/`](../gemma4-port/):

1. `01-confronto-architetturale.md` — DeepSeek V4 Flash vs Gemma 4 26B-A4B
2. `02-schema-quantizzazione.md` — ricetta IQ2_XXS/Q2_K/Q8_0 (~8.6 GiB)
3. `03-disk-kv-cache.md` — formato G4KV/G4SP e politiche di frontiera
4. `04-piano-implementazione.md` — milestone M0-M7

## Stato: M2 pre-validata (convertitore) + M1 + tokenizer + sessioni/G4SP

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
| API di sessione (`g4.h`) | decode incrementale con KV persistente: ring di `sliding_window` righe sui layer locali, storia piena sui globali | fatto |
| payload G4SP (`g4_session_save/load_payload`) | serializzazione dello stato per il disk KV (doc 03 §5), dtype f32 nel path di riferimento | fatto |
| `tests/g4_session_test.c` | incrementale == batch; save a metà (oltre il wrap del ring) → resume in sessione nuova → continuazione **bit-esatta**; reject di shape/ctx sbagliati | verde |
| `tools/g4-quantize.c` | convertitore safetensors→GGUF senza template: config da config.json, shape dagli header degli shard con **verifica di orientamento** (rifiuta layout trasposti), split del fuso gate_up, padding del down ai blocchi k-quant, profili `f32`/`q8`/`q4`, scrittura streaming, `--dry-run`/`--compare-tensor` | fatto (v1: no imatrix, single-thread) |
| `scripts/gen_synthetic_hf.py` + `make test-converter` | checkpoint HF **sintetico** ricostruito dai pesi toy (stessi nomi/layout di google/gemma-4-26B-A4B-it) → convertito → il motore rifà i logits JAX **esatti** (5e-6), sia in RAM sia in modalità mmap | verde |
| modalità frugale mmap (`g4_model_load_mmap`) | mmap del GGUF + dequantizzazione al volo per tensore/esperto; picco RAM < 1 GiB → il modello 26B gira anche su macchine da 12 GB (lento, streaming da SSD) | fatto |
| `tools/g4-run.c` | CLI di generazione: carica modello + tokenizer, encode del prompt, decode greedy/sampling; smoke test end-to-end di un modello convertito | fatto |
| `scripts/gen_logit_vectors.py` + `tests/g4_m2_test.c` | vettori top-k logprob dal modello vero via transformers e gate M2 (argmax + distanza logprob) | scritti, da eseguire sulla macchina con i pesi |

```sh
make test
```

Il GGUF giocattolo è generato dai moduli JAX veri del fork gemma:

```sh
cd ../../gemma && python3 g4ref/gen_toy_vectors.py ../ds4/gemma4/tests/vectors/toy.gguf
```

## Uso col modello reale (dopo la conversione)

```sh
# conversione HF -> GGUF q8 (~25 GiB)
./g4-quantize --hf /percorso/weights --out gemma4-q8.gguf --profile q8

# tabella tokenizer (una tantum)
python3 scripts/gen_tokenizer_vectors.py

# generazione (mmap frugale: gira anche con 12 GB di RAM)
./g4-run -m gemma4-q8.gguf -t tests/vectors/tok_table.bin -p "Ciao, come stai?" -n 64

# gate M2 contro i vettori transformers
python3 scripts/gen_logit_vectors.py --weights /percorso/weights
./g4_m2_test gemma4-q8.gguf tests/vectors/real_vectors.bin 0.1
```
