# g4 — motore di inferenza per Gemma 4 26B-A4B

Motore minimale in C11 per Gemma 4 26B-A4B con quantizzazione asimmetrica
degli esperti routed e disk KV cache, sul modello concettuale di ds4.

I documenti di progetto sono in [`../gemma4-port/`](../gemma4-port/):

1. `01-confronto-architetturale.md` — DeepSeek V4 Flash vs Gemma 4 26B-A4B
2. `02-schema-quantizzazione.md` — ricetta IQ2_XXS/Q2_K/Q8_0 (~8.6 GiB)
3. `03-disk-kv-cache.md` — formato G4KV/G4SP e politiche di frontiera
4. `04-piano-implementazione.md` — milestone M0-M7

## Stato: M0 (scaffolding)

| Modulo | Contenuto | Stato |
|---|---|---|
| `g4_quants.[ch]` | quantizzatori IQ2_XXS/Q2_K/Q4_K/Q8_0 (da `gguf-tools/quants.[ch]`, simboli rinominati) | fatto |
| `g4_dequant.c` | dequantizzazione di riferimento, convenzioni di decode identiche ai kernel ds4.c | fatto |
| `g4_gguf.[ch]` | lettore/scrittore GGUF v3 minimale (chiavi `gemma4.*`) | fatto |
| `g4_kvstore.[ch]` | header G4KV 48 byte, SHA1, eviction score | fatto (contenitore; store completo a M6) |
| `tests/g4_test.c` | round-trip quant (incluso il caso padding down 704→768), round-trip GGUF, header/eviction KVG | verde |

```sh
make test
```

Prossime milestone (doc 04 §4): M1 grafo forward CPU validato contro il
modello giocattolo JAX; M2 convertitore `g4-quantize` dai safetensors reali.
