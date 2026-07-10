# Fase 4 — Piano di implementazione: motore "g4" per Gemma 4 26B-A4B

Piano esecutivo per un motore di inferenza minimale, autocontenuto e
verticale su Gemma 4 26B-A4B, che implementa lo schema di quantizzazione
(doc 02) e il disk KV cache (doc 03). Filosofia ereditata da ds4: un modello,
un motore, GGUF propri, validazione contro logits di riferimento
(`README.md:40-44`).

---

## 1. Linguaggio, ambiente, dipendenze

**Linguaggio: C11**, autocontenuto, zero dipendenze esterne oltre libc
(stessa scelta ds4: `ds4.c` non linka nemmeno GGML, `README.md:46-57`).
Motivi: riuso diretto e legale del codice ds4/llama.cpp già nel fork
(`gguf-tools/quants.[ch]`, layout IQ2_XXS/Q2_K/Q8_0/Q4_K, `LICENSE` MIT con
notice GGML), stesso modello mentale, niente runtime da trascinare.

**Ambiente di sviluppo: WSL2 su Windows 10** (Hyper-V è già attivo sulla
macchina target). Toolchain Linux standard (gcc/clang + make), che è anche
l'ambiente dei build CUDA di ds4 (`Makefile`, target `cuda-generic`).
Percorso alternativo nativo MSVC: scartato per la v1 (il codice ds4 da cui si
attinge è POSIX; il costo di doppia portabilità non paga).

**Backend di calcolo**:

- **M0-M6: CPU** (AVX2 se disponibile). A differenza di ds4 — dove il CPU
  path è solo diagnostico perché il modello è 81-91 GiB (`README.md:1194-1206`)
  — qui i pesi q2 sono 8.6 GiB e ~1.43 GiB di esperti attivi per token: la
  CPU è un target di *esercizio* legittimo per la validazione e un uso
  interattivo lento ma reale.
- **M7: CUDA sotto WSL2**, solo se la macchina ha GPU NVIDIA (dato hardware
  non specificato nel brief: il piano non ne dipende fino a M7). Il porting
  concettuale parte dai kernel `ds4_cuda.cu` (dot quantizzati, softmax,
  RMSNorm); Metal non è applicabile su Windows.

**Dipendenze di tooling** (non di runtime): Python solo per generare i
vettori di riferimento (HF `transformers`/`tokenizers`, e la reference JAX
del fork gemma) — mai richiesto per eseguire il motore.

---

## 2. Collocazione e struttura dei file

Nuova directory top-level **`gemma4/`** in questo fork (branch
`claude/gemma4-ds4-quantization-9420u4`), codebase separata da `ds4.c` ma
nello stesso repo per riusare `gguf-tools/` e le convenzioni di test.

```text
gemma4/
  Makefile             # target: g4, g4-quantize, g4_test, (m7: cuda)
  README.md            # stato, comandi, gate di qualità
  g4.h                 # API pubblica stretta (engine/session, stile ds4.h)
  g4.c                 # engine: loader GGUF, grafo forward CPU, sampling
  g4_gguf.[ch]         # lettura/scrittura GGUF (chiavi gemma4.*, doc 02 §6)
  g4_quants.[ch]       # quant/dequant/dot: adattamento di gguf-tools/quants.[ch]
                       #   + kernel dot per IQ2_XXS/Q2_K/Q8_0/Q4_K (da ds4.c CPU path)
  g4_tokenizer.[ch]    # BPE SentencePiece-style (▁, byte-fallback, 262k vocab)
  g4_chat.[ch]         # template turni <|turn>, canale thinking, parser
                       #   <|tool_call>call:NOME{json}, escape <|"|>
  g4_kvstore.[ch]      # porting delle parti generiche di ds4_kvstore.[ch]
                       #   (SHA1, header, eviction, trailer hooks) + G4KV/G4SP
  g4_cli.c             # CLI interattiva (linenoise.c riusato dal repo)
  tools/
    g4-quantize.c      # convertitore safetensors -> GGUF quantizzato
    json_min.[ch]      # parser JSON minimale (config/tokenizer/index/safetensors header)
  tests/
    g4_test.c          # runner unico (stile ds4_test)
    vectors/           # vettori tokenizer + logits di riferimento
  scripts/
    gen_tokenizer_vectors.py   # da tokenizer.json HF (una tantum)
    gen_logit_vectors.py       # da transformers/JAX (una tantum, macchina utente)
```

Nel fork **gemma** (stesso branch): solo `g4ref/` con lo script che
istanzia un **modello giocattolo** Gemma4-MoE (2 layer, dim ridotte) con la
reference JAX (`gemma/gm/nn/gemma4/`), salva pesi random e logits attesi.
È il ponte di validazione descritto in §5.

**Convertitore senza template.** `deepseek4-quantize` richiede un GGUF
template per metadata/tokenizer (`gguf-tools/README.md:60-67`); noi non ne
abbiamo uno. `g4-quantize` costruisce tutto da fonti primarie:

- shape/dtype dei tensori: header JSON degli shard safetensors;
- iperparametri: `config.json` (già validato in Fase 1);
- tokenizer: vocab/merges/added_tokens estratti da `tokenizer.json` e
  incorporati come array di metadata GGUF (il motore non parsa mai JSON a
  runtime);
- pipeline per-tensore: mapping HF→GGUF e formati della ricetta doc 02 §3/§6,
  split del fuso `gate_up_proj`, padding 704→768 del down, dequant bf16 →
  quantizzazione con imatrix opzionale (riuso di `ds4q_quantize_chunk`).
  Flag ereditati: `--dry-run`, `--compare-tensor`, `--threads`, override per
  famiglia (`--experts`, `--routed-down`, ...).

---

## 3. Tokenizer (dettaglio, da `tokenizer.json` committato)

Accertato dai metadati: BPE con vocab 262'144, **514'906 merge**, normalizer
`" "→"▁"`, pre-tokenizer split-on-space `MergedWithPrevious`, decoder con
`ByteFallback` + `Fuse`, `byte_fallback=true`, 24 added token (i token
speciali `<|turn>`, `<|tool_call>`, ecc.). Implementazione C:

- tabelle vocab/merges caricate dal GGUF (già compilate dal convertitore);
- encode: greedy BPE per rank di merge su heap (equivalente all'encoder BPE
  di ds4 per DeepSeek), byte-fallback sui token `<0xNN>` per input fuori
  vocabolario;
- added/special token riconosciuti a monte del BPE (come i "DS4 protocol
  specials", `README.md:1251-1254`);
- gate di correttezza: identità con HF `tokenizers` su un corpus di vettori
  (§5, M3) — encode e decode round-trip.

---

## 4. Milestone e criteri di accettazione

Ordine scelto per avere **validazione numerica prima dei pesi reali** (M1
non dipende dal download da 48 GiB) e per replicare la scala di qualità ds4
(`QA_BEFORE_RELEASES.md`).

| # | Deliverable | Gate di accettazione | Dipendenze |
|---|---|---|---|
| **M0** | Scaffolding: `g4_gguf`, `g4_quants`, `g4_kvstore` (contenitore), `g4_test` | round-trip GGUF write→read bit-esatto; quant→dequant IQ2_XXS/Q2_K/Q4_K/Q8_0 con errore entro le tolleranze dei formati; header/eviction G4KV su file sintetici | nessuna (gira in questo container) |
| **M1** | Grafo forward CPU f32 completo (attn locale/globale K=V, MoE top-8, ramo denso, RoPE parziale, softcap) | logits identici (tolleranza f32) alla reference JAX sul **modello giocattolo** random del fork gemma; test per-layer (attn-only, moe-only) | fork gemma, niente pesi reali |
| **M2** | `g4-quantize` + build **bring-up Q8_0-everything** e **Q4_K routed** dai safetensors reali | `--compare-tensor` su campione per famiglia; caricamento e summary shape/metadata; primi logits reali vs vettori `transformers` su prompt corti | **download pesi 48 GiB (macchina utente)**; script `gen_logit_vectors.py` |
| **M3** | Tokenizer + template chat/tool | 100% match su vettori tokenizer (corpus multilingue + edge case ▁/byte-fallback/specials); render dei turni conforme a `chat_template.jinja` (da scaricare, doc 01 §9.3) | tokenizer.json (già in repo) |
| **M4** | Generazione end-to-end CLI (greedy + sampling top-k/top-p/min-p), prefill chunked | generazione coerente dal build Q4_K; gate deterministico stile ds4-eval q1..q4 (`README.md:608-627`) su 4 prompt fissi | M2, M3 |
| **M5** | Raccolta imatrix col motore + build **q2-imatrix finale** | delta di qualità q2 vs q4 entro la banda ds4 sull'harness adattato da `gguf-tools/quality-testing/`; niente regressioni sul gate M4 | M4 |
| **M6** | Disk KV cache G4KV/G4SP completo (frontiere, hit/miss, rewind-da-checkpoint, tool-map) | save/load round-trip bit-esatto dello stato; resume senza re-prefill; test di rewind con replay ≤ intervallo continued | M4 |
| **M7** | (opzionale/hardware-dipendente) backend CUDA WSL2; SSD streaming esperti; server HTTP OpenAI/Anthropic-compatibile con exact-replay dei tool call | parità di logits CPU/GPU su vettori; throughput; conformità API con client agentici | M5, M6; GPU NVIDIA |

Stima di massima dello sforzo relativo: M1 e M2 sono i macigni (grafo e
convertitore); M0/M3/M6 medi; M4 piccolo una volta passato M2; M5 dipende
dalla banda macchina (la raccolta imatrix è un job offline lungo, come in
ds4, `gguf-tools/README.md:50-53`); M7 è un progetto a sé.

---

## 5. Strategia di validazione (il punto più importante)

1. **Modello giocattolo JAX** (M1): il fork gemma istanzia
   `TransformerConfig` con `enable_moe=True`, 2 layer (1 locale + 1 globale),
   dim ridotte ma **stesse proprietà strutturali** (K=V globale, RoPE 0.25,
   GeGLU, top-k con rinormalizzazione, per_expert_scale, softcap); pesi
   random con seed fisso esportati in un mini-GGUF f32. Il motore C deve
   riprodurre i logits JAX. Questo isola i bug di grafo da quelli di
   conversione/quantizzazione **prima** di toccare i 48 GiB.
2. **Vettori ufficiali** (M2+): `gen_logit_vectors.py` con `transformers`
   (v5.5+, `Gemma4ForConditionalGeneration`) genera coppie prompt→top-k
   logprob sul checkpoint bf16 reale; equivalente dei
   `tests/test-vectors` ds4 (`README.md:1219-1237`). Nota RAM: servono
   ~52 GiB per il forward bf16 su CPU — se la macchina utente non li ha,
   fallback con offload a disco di `accelerate` (lento ma una tantum) o
   generazione su una macchina cloud temporanea; i vettori sono piccoli e
   si committano.
3. **Tolleranze quantizzate**: per i build Q8/Q4/q2 il confronto è per
   token-match del greedy e distanza sui top-k logprob (soglie calibrate
   sul comportamento ds4 tra le sue varianti), non per uguaglianza esatta.
4. **Gate deterministici permanenti** (da M4): 4 prompt fissi con conteggi
   di token attesi, replicando la tabella di regressione ds4
   (`README.md:608-627`); girano in CI locale (`make test`).

---

## 6. Riuso dal repo ds4 (inventario esplicito)

| Sorgente | Riuso | Modo |
|---|---|---|
| `gguf-tools/quants.[ch]` | layout blocchi, quantizzatori IQ2_XXS/Q2_K/Q4_K/Q8_0, imatrix hook | copia in `g4_quants.[ch]`, si aggiungono i dot-kernel |
| `ds4.c` (CPU path) | dot quantizzati AVX2, RMSNorm, softmax, sampling (`ds4_sample_logits`) | estrazione mirata |
| `ds4_kvstore.[ch]` | SHA1, header 48B, refresh/touch/evict, trailer hooks | copia con rinomina, payload sostituito da G4SP |
| `rax.[ch]` | radix tree per la tool-map exact-replay | uso diretto |
| `linenoise.[ch]` | CLI interattiva | uso diretto |
| `gguf-tools/deepseek4-quantize.c` | struttura del convertitore (policy per famiglia, `--compare-tensor`, workers) | riferimento, riscritto senza template |
| `gguf-tools/imatrix/`, `quality-testing/` | pipeline calibrazione e scoring | adattamento (render col template Gemma) |
| Grafo attenzione/MoE, tokenizer, template, server | **niente riuso**: specifici DeepSeek | riscrittura (docs 01-03) |

---

## 7. Fuori scope per la v1

Multimodale (vision tower esclusa, doc 02 §1), inferenza distribuita,
speculative/MTP (Gemma 4 26B-A4B non pubblica un modulo MTP), thinking-budget
speciali (si parsa il canale `<|channel>thought` ma senza modalità "Max"
DeepSeek-style), batch > 1, Windows nativo MSVC.

---

## 8. Rischi principali

| Rischio | Impatto | Mitigazione |
|---|---|---|
| Orientamento/layout dei tensori HF diverso dall'atteso | conversione errata | M1 prima dei pesi reali; header safetensors letti e confrontati con le shape attese in `g4-quantize` (fail-fast); `--compare-tensor` |
| RAM insufficiente per i vettori di riferimento bf16 | blocco M2 | offload `accelerate`, o vettori generati una tantum altrove e committati |
| Divergenza `approx_max_k` vs top-8 esatto | mismatch logits su token border-line | i vettori di riferimento si generano con `transformers` (top-k esatto), non con la pipeline TPU |
| Hardware GPU ignoto | M7 incerto | M0-M6 CPU-only per costruzione |
| Dettagli del template chat non ancora scaricati | render M3 | `chat_template.jinja` + `generation_config.json` da aggiungere a `hf-metadata/` (comando in doc 01 §9.3) prima di M3 |

---

## 9. Primo passo esecutivo

M0 non ha dipendenze esterne: scaffolding `gemma4/` con GGUF r/w, port dei
quantizzatori con test di round-trip e contenitore G4KV su payload sintetici.
È il punto di partenza naturale ed è interamente verificabile senza pesi.
