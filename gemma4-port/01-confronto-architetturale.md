# Fase 1 — Confronto architetturale: DeepSeek V4 Flash (ds4) vs Gemma 4 26B-A4B

Documento di ricognizione per il porting concettuale dello schema ds4
(quantizzazione asimmetrica dei soli esperti routed + KV cache come cittadino
di prima classe su SSD) sull'architettura di Gemma 4 26B-A4B.

**Fonti primarie:**

- Codice ds4 (questo repo): `ds4.c`, `ds4.h`, `ds4_kvstore.h`, `ds4_ssd.h`,
  `README.md`, `MODEL_CARD.md`, `gguf-tools/` (`deepseek4-quantize.c`,
  `quants.c/h`).
- Reference implementation JAX di Gemma 4 (fork di google-deepmind/gemma):
  `gemma/gm/nn/gemma4/` (`_gemma4.py`, `_config.py`, `_modules.py`, `_moe.py`,
  `_transformer.py`, `_layers.py`) e `gemma/gm/text/_tokenizer.py`.

- Metadati Hugging Face di `google/gemma-4-26B-A4B-it` in
  `gemma4-port/hf-metadata/` (`config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `model.safetensors.index.json`).

**Stato:** completo. I punti inizialmente marcati **[HF-PENDING]** sono stati
chiusi con i metadati HF (vedi §9); l'unico residuo minore è il file
`chat_template.jinja` (§9.3).

---

## 1. Vista d'insieme

| Proprietà | DeepSeek V4 Flash | Gemma 4 26B-A4B |
|---|---|---|
| Parametri totali | 284B (`MODEL_CARD.md:16`) | ~25.2B calcolati (§5.3), "26B" nominali |
| Parametri attivi/token | 13B (`MODEL_CARD.md:16`) | ~3.8B calcolati (§5.3), "A4B" nominali |
| Layer | 43 (`ds4.c:180`) | 30 (`_gemma4.py:31`) |
| Hidden size (d_model) | 4096 (`ds4.c:181`) | 2816 (`_gemma4.py:260`) |
| Vocab | 129280 (`ds4.c:182`) | 262144 (`_gemma4.py:259`) |
| Attenzione | MLA/latente + compressione temporale (CSA/HCA) + indexer | GQA ibrida local-sliding/global, K=V sui layer globali |
| Q heads | 64 × 512 dim (`ds4.c:183,185`) | 16 × 256 dim (locale) / 16 × 512 dim (globale) (`_gemma4.py:262-263`, `_modules.py:508-514`) |
| KV heads | 1 latente × 512 (`ds4.c:184-186`) | 8 × 256 (locale), 2 × 512 (globale) (`_gemma4.py:264,266,271`) |
| Esperti routed | 256, top-6 (`ds4.c:191-192`) | 128, top-8 (`_gemma4.py:282,284`) |
| Esperti condivisi | 1 shared expert MoE-style (`ds4.c:193`) | nessuno; in cambio c'è un **ramo MLP denso parallelo** per layer (`_modules.py:562-591`) |
| FFN dim esperto | 2048 (`ds4.c:194`) | 704 (`_gemma4.py:283`) |
| Attivazione FFN | SwiGLU con clamp (`ds4.c:205`, `swiglu_clamp_exp`) | **GeGLU**: `gelu(x1) * x2` (`_moe.py:346`, `_modules.py:453`) |
| Layer MoE | tutti tranne i primi `n_hash_layer=3` densi (`ds4.c:195`) | **tutti e 30** (`_transformer.py:189`, `enable_moe` passato a ogni Block) |
| Context max | 1M token (`MODEL_CARD.md:16`) | **262144 (256k)** (`hf-metadata/config.json:68`, `max_position_embeddings`) |
| Softcap logits finali | no | tanh cap a 30.0 (`_gemma4.py:265`, `_transformer.py:334-336`) |
| Embeddings | separate da output head (`output.*` in `deepseek4-quantize.c:1009`) | **tied**: `decode = x @ input_embedding_table.T` (`_modules.py:127-137`) |
| Extra strutturali | Manifold-Constrained Hyper-Connections, n_hc=4 (`ds4.c:200`, `MODEL_CARD.md:67`) | `skip_scale` scalare per blocco (`_modules.py:499,662`); norm post-attn e post-ffw (`_gemma4.py:267-268`) |
| Licenza | MIT (`MODEL_CARD.md:231`) | Apache 2.0 (header dei sorgenti JAX; per i pesi, dichiarata Apache 2.0 nel brief di progetto — il file LICENSE del repo HF non è tra i metadati scaricati) |

Nota dimensioni: 25.2B × 2 byte (bf16) ≈ 50.4 GB, coerente con i ~52 GB dei
pesi HF citati nel brief del progetto.

---

## 2. Attenzione

### 2.1 DeepSeek V4 Flash: MLA + compressione temporale + indexer

Per ds4 l'attenzione è il pezzo più esotico (`MODEL_CARD.md:23-64`):

- KV **latente a 1 testa** da 512 dim (`n_head_kv=1`, `n_head_dim=512`,
  `ds4.c:184-186`), query e output proiettati via LoRA
  (`n_lora_q=1024`, `n_lora_o=1024`, `ds4.c:189-190`; tensori
  `attn_q_a/attn_q_b`, `attn_output_a/attn_output_b` in
  `deepseek4-quantize.c:930-936`).
- Ogni layer tiene una **finestra raw di 128 token** (`n_swa=128`, `ds4.c:196`).
- Layer 0-1: solo finestra raw. Layer pari ≥2: KV compressa **ratio-4** con
  stream **indexer** separato (64 teste × 128 dim, top-k 512,
  `ds4.c:197-199`) che seleziona le righe compresse visibili. Layer dispari
  ≥3: KV compressa **ratio-128** senza indexer (`MODEL_CARD.md:32-50`).
- Tensori dedicati: `attn_compressor_{kv,gate,norm,ape}`,
  `indexer_compressor_*`, `indexer.attn_q_b`, `indexer.proj`
  (`deepseek4-quantize.c:937-946`).

È compressione **sull'asse temporale** (più posizioni → una riga KV), motivo
per cui ds4 può esporre 1M di contesto: ~26 GB di KV a 1M token, di cui ~22 GB
di indexer (`README.md:829-832`).

### 2.2 Gemma 4 26B-A4B: GQA ibrida local/global

Niente MLA, niente compressione temporale, niente indexer. Lo schema è
un'evoluzione del pattern Gemma 3:

- **Pattern di layer**: `(LOCAL_SLIDING ×5, GLOBAL) ×5` → 25 layer locali,
  5 globali (indici 0-based 5, 11, 17, 23, 29) (`_gemma4.py:247-257`,
  `_config.py:41-52`).
- **Layer locali**: GQA 16 Q heads / 8 KV heads × 256 dim, K e V proiettati
  da un unico `kv_einsum` di shape `(2, 8, 2816, 256)` (`_modules.py:233-235`),
  sliding window **1024** (`_gemma4.py:276`; maschera in `_modules.py:38-52`),
  RoPE base 10'000 su tutte le dim (`local_rope_proportion=1.0`,
  `_gemma4.py:273-277`).
- **Layer globali**: 16 Q heads ma solo **2 KV heads × 512 dim**
  (`num_global_kv_heads=2`, `global_key_size=512`, `_gemma4.py:266,271`) e
  soprattutto **`k_eq_v=True`**: un solo `k_einsum` di shape `(2, 2816, 512)`
  produce un'unica proiezione usata sia come K sia come V
  (`_modules.py:228-231, 277-283`). Attenzione: dopo la proiezione condivisa,
  K passa per `key_norm` (RMSNorm **con** scale) + RoPE, V passa per
  `value_norm` (RMSNorm **senza** scale) e niente RoPE (`_modules.py:236-238,
  282-295`) — quindi i valori cached di K e V **non sono identici** anche se
  la matrice dei pesi è una sola.
- RoPE globale: base 1'000'000 e `rope_proportion=0.25` → solo il 25% delle
  512 dim ruotato (`_gemma4.py:273-274`); è il segnale tipico del design
  long-context.
- QK-norm con scale su Q e K (`qk_norm_with_scale=True`, `_gemma4.py:269`,
  `_modules.py:236-237`); niente softcap sui logit di attenzione
  (`attn_logits_soft_cap=None`, `_gemma4.py:275`).
- Nessuna condivisione di KV tra layer per questo modello
  (`kv_cache_sharing_config` non impostato → default `None`,
  `_config.py:115`, `_gemma4.py:258-296`).

### 2.3 Conseguenza pratica per il porting

L'idea ds4 "la KV cache è un cittadino di prima classe su disco" si trasferisce
bene, ma i **contenuti** dello stato per layer sono completamente diversi:

- ds4 serializza per layer: righe raw della finestra 128, righe KV compresse +
  frontiere del compressor, righe indexer + frontiere (`README.md:1112-1121`).
- Gemma 4 richiede per layer solo: righe K/V della finestra (locale, bounded a
  1024) oppure K/V globali (unbounded), più le posizioni assolute per la
  maschera sliding (`_modules.py:399-419`: cache = `k`, `v`, `end_index`,
  `positions`). Non esistono "frontiere" di compressione: lo stato è
  puramente per-token.

---

## 3. KV cache: forma e dimensioni per token

Layout della cache di Gemma 4 (`_modules.py:399-419`, ring buffer con
scrittura modulo `cache_size`, `_modules.py:301-317`):

| Componente | Shape per layer | Byte/token (bf16) |
|---|---|---|
| Layer locale (×25) | K,V: `[B, cache, 8, 256]` | 8 KiB/token, **bounded**: serve tenere solo gli ultimi 1024 token → ≤ 8 MiB/layer |
| Layer globale (×5) | K,V: `[B, cache, 2, 512]` | 4 KiB/token (2 KiB K + 2 KiB V), **unbounded** |
| positions | `[B, cache]` int32 | 4 B/token/layer |

Totali (batch 1):

- **Stato bounded** (25 layer locali): ≤ 200 MiB indipendentemente dal
  contesto.
- **Stato unbounded** (5 layer globali): 20 KiB/token → 2.5 GiB a 128k token.
  Asintoticamente comparabile ai ~26 KiB/token di DeepSeek V4 Flash a 1M
  (`README.md:829-832`): Gemma 4 ottiene con SWA + pochi KV head globali ciò
  che DeepSeek ottiene con la compressione temporale.
- Ottimizzazione possibile (da valutare in Fase 3): sui layer globali K e V
  derivano dalla stessa proiezione; salvando su disco la proiezione pre-norm
  (2 KiB/token/layer) e riapplicando `key_norm`/`value_norm`/RoPE al load si
  dimezza lo stato unbounded a costo di ricomputo. In RAM conviene tenerli
  separati.

---

## 4. Routing MoE

### 4.1 DeepSeek V4 Flash (per riferimento)

- 256 esperti routed, 6 attivi, **1 esperto condiviso** sempre attivo
  (`ds4.c:191-193`).
- Router con bias di selezione (`exp_probs_b.bias`,
  `deepseek4-quantize.c:954`), gruppi di esperti
  (`deepseek4.expert_group_count`, `ds4.c:2030`), scala dei pesi degli esperti
  `expert_weight_scale=1.5` (`ds4.c:204`).
- Primi 3 layer densi (`n_hash_layer=3`, `ds4.c:195`).

### 4.2 Gemma 4 26B-A4B

Ogni layer ha **due rami FFN paralleli** sommati (`_modules.py:674-693`):

```
out = post_ffw_norm( post_ffw2_norm(mlp2_denso(x)) + post_ffw1_norm(moe(x)) )
```

1. **Ramo denso condiviso** (`mlp2`): FeedForward GeGLU 2816 → 2112 → 2816
   (`moe_dense_hidden_dim=2112`, `_gemma4.py:285`; setup in
   `_modules.py:562-574`). È l'equivalente funzionale dello shared expert di
   DeepSeek: capacità sempre attiva, per ogni token.
2. **Ramo MoE** (`mlp`): 128 esperti, top-8, `expert_dim=704`
   (`_gemma4.py:282-284`; `MoERagged` in `_moe.py:261-407`).

Algoritmo di routing (`_moe.py:301-310, 381-407`):

- input del router = `router_norm(x_non_normalizzato)` (RMSNorm senza scale)
  moltiplicato per `rsqrt(2816) * router_scale` (`router_scale` è un vettore
  appreso di dim 2816) — nota: il router riceve l'attivazione **pre**
  `pre_ffw_norm`, passata esplicitamente come `unnormalized_x`
  (`_modules.py:684`, `_moe.py:394-404`);
- logits = `x @ W_router` con `W_router` di shape `(2816, 128)`
  (`_moe.py:274-276`);
- **softmax su tutti i 128** esperti, poi top-8 (`jax.lax.approx_max_k`),
  poi **rinormalizzazione dei pesi dei soli esperti scelti**
  (`_moe.py:301-310`, `_renormalization_factor` in `_moe.py:23-35`). Schema
  softmax-then-topk in stile Gemma/GShard, diverso dal sigmoid+bias di
  DeepSeek;
- output di ogni esperto moltiplicato per `per_expert_scale[e]` (vettore
  appreso di dim 128, `_moe.py:289-293, 357-364`).

Nota implementativa: `jax.lax.approx_max_k` è un top-k **approssimato** (TPU).
In un motore C il top-8 esatto è la scelta naturale; l'eventuale divergenza
dai logits ufficiali va verificata in fase di validazione. Il config HF
conferma gli iperparametri: `num_experts=128`, `top_k_experts=8`,
`enable_moe_block=true`, `moe_intermediate_size=704`
(`hf-metadata/config.json:70-92`); attivazione `gelu_pytorch_tanh`
(approssimazione tanh della GELU, `config.json:31`) e
`use_double_wide_mlp=false` (`config.json:95`).

### 4.3 Peso relativo dei rami

Parametri per layer: ramo MoE ≈ 761.2M (128 esperti × 5.947M), ramo denso
mlp2 ≈ 17.8M, attenzione ≈ 34.6M (locale) / 49.1M (globale), router ≈ 0.36M.
**Gli esperti routed sono il 90.6% del modello** (22.84B su 25.2B): la
precondizione dello schema ds4 ("i routed dominano lo spazio del modello",
`README.md:100-103`) vale per Gemma 4 ancora più che per DeepSeek Flash.

---

## 5. Tensori degli esperti routed: nomi e shape

### 5.1 ds4 / DeepSeek (GGUF)

Il quantizzatore riconosce come esperti routed esattamente
(`deepseek4-quantize.c:881-896`):

```
blk.N.ffn_gate_exps.weight   → w1 (gate)
blk.N.ffn_down_exps.weight   → w2 (down)
blk.N.ffn_up_exps.weight     → w3 (up)
```

Shared expert e router restano fuori: `ffn_{gate,up,down}_shexp.weight`,
`ffn_gate_inp.weight`, `exp_probs_b.bias` (`deepseek4-quantize.c:949-954`,
`is_shared_expert()` a `deepseek4-quantize.c:1003-1005`).

### 5.2 Gemma 4 (parametri Flax e nomi safetensors HF)

Per ogni `layer_i` (i = 0..29), nel param-tree Flax (`_moe.py:273-299`,
`_modules.py:496-591`):

| Parametro Flax | Shape | Ruolo |
|---|---|---|
| `layer_i/mlp/gating_einsum/w` | `(128, 2, 704, 2816)` | **gate+up fusi** di tutti gli esperti (indice 2: k=0 gate, k=1 up; usato come `[E, F, 2H]` in `ragged_dot`, `_moe.py:326-346`) |
| `layer_i/mlp/linear/w` | `(128, 704, 2816)` | **down** di tutti gli esperti |
| `layer_i/mlp/router_logits/w` | `(2816, 128)` | router (NON quantizzare) |
| `layer_i/mlp/router_scale` | `(2816,)` | scala input router (NON quantizzare) |
| `layer_i/mlp/per_expert_scale` | `(128,)` | scala output per esperto (NON quantizzare) |
| `layer_i/mlp2/gating_einsum` | `(2, 2112, 2816)` | gate+up del ramo denso condiviso (`_modules.py:441-448`) |
| `layer_i/mlp2/linear` | `(2112, 2816)` | down del ramo denso condiviso |

Per singolo esperto: gate `(704, 2816)`, up `(704, 2816)`, down `(704, 2816)`
= 5'947'392 parametri (~11.9 MB bf16, contro i ~50 MB/esperto di Flash: la
granularità di streaming/cache è ~4× più fine).

Differenza strutturale importante rispetto a DeepSeek: in Gemma 4 gate e up
sono **un solo tensore fuso** con asse `k∈{0,1}`. Per applicare bit-width
diversi a gate/up vs down (schema ds4) non serve separarli: gate+up stanno
già nello stesso tensore e riceverebbero comunque lo stesso formato; il down
è un tensore separato. Attenzione ai vincoli di blocco dei formati k-quant:
la contrazione di gate/up avviene su 2816, che è multiplo di 256 (`QK_K`,
`quants.c:34`), ma quella del down avviene su 704, che **non** lo è
(704 = 2×256 + 192): per quantizzare il down con Q2_K servono padding a 768
o un layout trasposto — la scelta è documentata nella Fase 2 (doc 02).

Nomi confermati nei safetensors HF (`hf-metadata/model.safetensors.index.json`,
1013 tensori, `total_size` = 51'611'872'412 byte = 48.07 GiB, dtype bf16 da
`config.json:9`), per ogni layer `i` sotto il prefisso
`model.language_model.layers.i.`:

| Tensore HF | Corrispondenza Flax | Note |
|---|---|---|
| `experts.gate_up_proj` | `mlp/gating_einsum/w` | **fuso** gate+up, ×30 layer, senza suffisso `.weight` |
| `experts.down_proj` | `mlp/linear/w` | ×30 layer |
| `router.proj.weight` | `mlp/router_logits/w` | ×30 |
| `router.scale` | `mlp/router_scale` | ×30 |
| `router.per_expert_scale` | `mlp/per_expert_scale` | ×30 |
| `mlp.gate_proj.weight` / `mlp.up_proj.weight` / `mlp.down_proj.weight` | `mlp2/*` | ramo denso: in HF gate e up sono **separati** |
| `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.o_proj.weight` | `attn/*_einsum` | ×30 |
| `self_attn.v_proj.weight` | `attn/kv_einsum` (parte V) | **solo ×25**: sui 5 layer globali non esiste, conferma K=V |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | `query_norm`/`key_norm` | nessun parametro per `value_norm` (RMSNorm senza scale) |
| `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`, `post_feedforward_layernorm` | le 7 RMSNorm del blocco | mapping preciso MoE/denso da verificare a livello numerico in implementazione |
| `layer_scalar` | `skip_scale` | ×30 |

Fuori dai layer: `model.language_model.embed_tokens.weight` (tied, nessun
`lm_head` separato, coerente con `tie_word_embeddings=true`,
`config.json:91,99`) e `model.language_model.norm.weight` (final norm). I
tensori `model.vision_tower.*` e `model.embed_vision.*` (~1.1 GiB) riguardano
l'encoder visivo. L'index non riporta le shape: l'orientamento esatto
(convenzione torch `[out, in]` per i `.weight`) va confermato leggendo gli
header dei singoli shard safetensors in fase di conversione.

### 5.3 Verifica di coerenza dei conteggi

- Embeddings (tied): 262144 × 2816 = 0.738B.
- Esperti routed: 30 × 128 × 5.947M = **22.84B**.
- Ramo denso mlp2: 30 × 17.84M = 0.535B.
- Attenzione: 25 × 34.6M + 5 × 49.1M = 1.11B.
- Router + norm + scale: ≈ 0.012B.
- **Totale ≈ 25.2B** ✓ ("26B").
- Attivi/token: embeddings + attn + mlp2 + router + 8/128 degli esperti
  ≈ 0.74 + 1.11/30·layer… ≈ **3.8B** ✓ ("A4B").

---

## 6. Mappatura dello schema di quantizzazione ds4 su Gemma 4 (anteprima Fase 2)

Ricetta ds4 per Flash (`README.md:100-103`, `gguf-tools/README.md:69-97`,
formati in `quants.c:39-75`):

| Famiglia | Formato | bit/peso |
|---|---|---|
| routed w1/w3 (gate/up) | IQ2_XXS (blocchi 256, 66 B) | 2.0625 |
| routed w2 (down) | Q2_K (blocchi 256, 84 B) | 2.625 |
| proiezioni attenzione | Q8_0 (blocchi 32, 34 B) | 8.5 |
| shared experts | Q8_0 | 8.5 |
| output head | Q8_0 | 8.5 |
| compressor/indexer | F16 | 16 |

Traduzione naturale su Gemma 4 (i numeri e la scelta definitiva dei bit-width
sono materia della Fase 2):

| Tensore Gemma 4 | Analogo ds4 | Candidato |
|---|---|---|
| `mlp/gating_einsum` (gate+up routed) | `ffn_gate_exps`+`ffn_up_exps` | IQ2_XXS |
| `mlp/linear` (down routed) | `ffn_down_exps` | Q2_K |
| `mlp2/*` (ramo denso) | `ffn_*_shexp` (shared expert) | Q8_0 |
| `attn/{q,kv,k,attn_vec}_einsum` | proiezioni attenzione | Q8_0 |
| `router_logits`, `router_scale`, `per_expert_scale`, tutte le RMSNorm, `skip_scale` | routing/norm | F32/F16 |
| `input_embedding` (tied, fa anche da output head) | `token_embd` + `output` | Q8_0 (da valutare: è sia embedding sia LM head) |

Stima grezza della dimensione risultante con questa ricetta:
esperti ≈ 22.84B × ~2.25 bit medi ≈ 6.4 GB; resto ≈ 2.4B × ~8.5 bit ≈ 2.5 GB;
**totale ≈ 9 GB** (dai ~50 GB bf16). Anche il rapporto di compressione è più
favorevole che su Flash proprio perché la frazione routed è più alta.

Punto aperto per la Fase 2: IQ2_XXS richiede una imatrix (
`ds4q_requires_imatrix`, `quants.c:54`; fallback sintetico weight-energy in
`gguf-tools/README.md:111-124`). ds4 raccoglie la imatrix con il runtime
stesso (`gguf-tools/imatrix/`): per Gemma 4 servirà l'equivalente (o il
fallback sintetico per la prima iterazione).

---

## 7. Tokenizer e tool-calling

| Aspetto | DeepSeek V4 Flash | Gemma 4 |
|---|---|---|
| Modello tokenizer | BPE proprietario, vocab 129280 (`ds4.c:182`) | **SentencePiece** `tokenizer_gemma4.model`, VERSION=4, vocab 262144 (`_tokenizer.py:477-487`, `_gemma4.py:259`); su HF classe `GemmaTokenizer` backend `tokenizers` con `tokenizer.json` da ~32 MB (`hf-metadata/tokenizer_config.json:3,72`) |
| Sorgente template | renderer Python ufficiale `encoding_dsv4.py`, niente Jinja (`MODEL_CARD.md:139-146`) | formato `dialog.Format.GEMMA4` (`_tokenizer.py:486`); lato HF il formato di output è codificato nel `response_schema` regex di `tokenizer_config.json:25-65`; il file `chat_template.jinja` (input) è separato e non ancora scaricato (§9.3) |
| Token di turno | `<｜User｜>`, `<｜Assistant｜>`, BOS/EOS dedicati (`MODEL_CARD.md:147-158`) | `<|turn>` = 105, `<turn|>` = 106, BOS=2, EOS=1, PAD=0 (`_tokenizer.py:138-166`); EOS di generazione doppio: `[1, 106]` (`config.json:13-16`) |
| Thinking | `<think>`/`</think>` + modalità Max (`MODEL_CARD.md:90-99`) | canale dedicato: `<|channel>thought\n ... <channel|>` prima del contenuto (regex in `tokenizer_config.json:64`); esiste anche un `think_token` `<|think|>` (`tokenizer_config.json:71`) |
| Tool-calling | **DSML** testuale, con canonicalizzazione + exact-replay map nel server (`README.md:771-801`) | blocchi `<|tool_call>call:NOME{argomenti JSON}<tool_call|>` (regex `tokenizer_config.json:38-48`, parser dedicato `x-parser: gemma4-tool-call`), risposte in `<|tool_response>...<tool_response|>`, definizioni in `<|tool>...<tool|>`, token di escape `<|\"|>` (`tokenizer_config.json:12-15,66-70`) |

Implicazione: un equivalente del parser DSML **serve**, ma è più semplice.
Il problema che ds4 risolve con l'exact-replay DSML (il client rimanda la
history in JSON normalizzato e il re-render deve combaciare byte-per-byte col
KV checkpoint, `README.md:771-792`) esiste identico anche con Gemma 4 — è una
proprietà delle API stateless, non del formato — quindi la parte di design
"tool id → blocco sampled esatto" va ripresa; cambia solo la sintassi da
parsare: `<|tool_call>call:NOME{...}<tool_call|>` con argomenti JSON e token
di escape `<|"|>`, invece di DSML. Il rendering lato input (system prompt,
definizioni tool in `<|tool>...<tool|>`) va confermato da
`chat_template.jinja` (§9.3).

Bug noto citato nei sorgenti: i modelli nano (E2B/E4B) a volte emettono
`<eos>` invece di `<|tool_response>` (`gm/tools/_manager.py:64-67`) — da
tenere presente nel parser, anche se il 26B-A4B non è indicato come affetto.

---

## 8. Cosa si porta e cosa si riprogetta (sintesi)

**Si trasferisce quasi invariato da ds4:**

- Quantizzazione asimmetrica dei soli esperti routed (la frazione routed in
  Gemma 4 è persino maggiore: 90.6% vs ~85% di Flash).
- SSD streaming degli esperti: cache RAM di esperti interi + miss da file
  (`README.md:180-268`, `ds4_ssd.h`); con esperti da ~12MB (bf16) o ~1.6MB
  (2-bit) la granularità è più fine e il costo di un miss minore.
- L'infrastruttura disk-KV a livello file: header tipo KVC + testo renderizzato
  come chiave (SHA1) + payload di stato + tool-map opzionale
  (`README.md:1027-1096`, `ds4_kvstore.h`), frontiere di salvataggio
  cold/continued/evict/shutdown con trim e align (`README.md:1139-1156`).

**Si riprogetta da zero:**

- Il grafo di attenzione: GQA local/global con sliding window 1024, K=V sui
  globali, QK-norm, RoPE parziale (25%) sui globali — nessun riuso del codice
  MLA/compressor/indexer di ds4.
- Il payload KV su disco: niente righe compresse né frontiere; per layer
  servono (a) finestra locale bounded in ordine logico, (b) K/V globali
  unbounded, (c) posizioni. La struttura DSV4 (`README.md:1091-1121`,
  `ds4.h:304-309`) va sostituita con un formato specifico (Fase 3).
- Il routing MoE: softmax→top-8→renorm + `per_expert_scale` + router con
  RMSNorm/scale dedicati, contro sigmoid+bias+gruppi di DeepSeek.
- Il renderer di chat/tool: SentencePiece + template Gemma 4 + parser JSON dei
  tool-call, contro BPE DeepSeek + DSML.
- Il ramo denso per layer (`mlp2`) non ha equivalente diretto in ds4 (lo
  shared expert di DeepSeek è nel gruppo MoE); nel grafo va trattato come una
  seconda FFN sempre attiva i cui output si sommano a quelli MoE prima della
  post-norm.

---

## 9. Punti [HF-PENDING]: esito dopo l'acquisizione dei metadati

I metadati sono in `gemma4-port/hf-metadata/`. Esito punto per punto:

1. **Context window**: `max_position_embeddings = 262144` (256k),
   `config.json:68`. RoPE: sliding `theta=10000` tipo `default`; full
   `theta=1000000`, `partial_rotary_factor=0.25`, tipo `proportional`
   (`config.json:79-89`) — coerente col codice JAX, nessuno scaling YaRN.
2. **Tensori safetensors**: nomi confermati e tabulati in §5.2; dtype bf16
   (`config.json:9,25`); `total_size` 48.07 GiB. Le **shape** non sono
   nell'index: verifica finale dagli header degli shard in fase di
   conversione.
3. **Chat template**: parzialmente chiuso. Il formato di output (thinking come
   canale `<|channel>thought`, tool call `call:NOME{json}`, terminatori) è nel
   `response_schema` (`tokenizer_config.json:25-65`). **Residuo**: il file
   `chat_template.jinja` (in transformers v5 è separato dal
   `tokenizer_config.json`) per il rendering lato input — da scaricare insieme
   a `generation_config.json` prima della Fase 4:
   `curl.exe -L -H "Authorization: Bearer $env:HF_TOKEN" -o chat_template.jinja https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/chat_template.jinja`
4. **Routing HF**: `num_experts=128`, `top_k_experts=8`,
   `enable_moe_block=true` (`config.json:72,92,26`); nessun softcap
   d'attenzione nel config (solo `final_logit_softcapping=30.0`,
   `config.json:28`). Confermati anche `attention_k_eq_v=true`,
   `global_head_dim=512`, `num_global_key_value_heads=2`,
   `sliding_window=1024`, `rms_norm_eps=1e-06`, eos doppio `[1, 106]`.
5. **Dimensione totale**: 51'611'872'412 byte (48.07 GiB), inclusi ~1.1 GiB
   di vision tower — in linea con la stima §5.3 per la sola parte testo.

Unico residuo: `chat_template.jinja` + `generation_config.json` (punto 3),
necessari solo per il renderer di chat della Fase 4.
