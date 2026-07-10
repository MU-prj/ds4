# Fase 2 — Schema di quantizzazione asimmetrica per Gemma 4 26B-A4B

Progettazione della ricetta di quantizzazione, derivata dallo schema ds4 per
DeepSeek V4 Flash (`README.md:100-103`: routed up/gate a IQ2_XXS, routed down
a Q2_K, tutto il resto ad alta precisione) e adattata alla struttura reale di
Gemma 4 26B-A4B accertata in Fase 1 (`01-confronto-architetturale.md`).

Prerequisiti numerici (Fase 1, §5.3): modello testo ≈ 25.23B parametri bf16
(47.0 GiB; 48.07 GiB col vision tower, `hf-metadata/model.safetensors.index.json`),
di cui **22.84B (90.6%) negli esperti routed**.

---

## 1. Perimetro

**Decisione P1 — motore text-only.** I tensori `model.vision_tower.*` e
`model.embed_vision.*` (~1.07 GiB bf16) non vengono convertiti né caricati.
Analogo alla scelta di ds4 (motore testuale puro). L'eventuale supporto
multimodale è fuori scope per la v1.

**Decisione P2 — quantizzazione aggressiva solo sugli esperti routed.**
Identica filosofia ds4: gli esperti routed sono la quasi totalità dello
spazio (90.6% qui, contro ~85% su Flash), ogni token ne usa 8/128, e la
ridondanza del MoE assorbe l'errore di quantizzazione molto meglio dei
componenti sempre attivi. Tutto ciò che è denso/attivo per ogni token resta
ad alta precisione.

---

## 2. Il vincolo di blocco sul down (704 non divisibile per 256)

I formati k-quant/i-quant usati da ds4 lavorano su blocchi di `QK_K = 256`
pesi lungo la dimensione di contrazione della matmul (`quants.c:34`,
`ds4q_row_size` a `quants.c:1032-1037` fallisce se `ne % block_size != 0`).

- gate/up: contrazione su d_model = 2816 = 11 × 256 → **ok senza modifiche**;
- down: contrazione su expert_dim = 704 = 2 × 256 + 192 → **non compatibile**.

Alternative valutate:

| Opzione | Descrizione | Costo | Contro |
|---|---|---|---|
| **A. Zero-padding 704→768** (scelta) | ogni riga del down estesa con 64 zeri; 768 = 3×256 | +216.6 MiB (+9.1% sul down, +2.5% sul totale) | byte "sprecati" su disco/RAM |
| B. Layout trasposto | down salvato con contrazione sull'asse 2816, kernel column-accumulate (axpy) dedicato | 0 byte | kernel diverso dal dot-row usato ovunque; accumulare f32[2816] (11 KiB) per token; più codice da validare |

**Decisione P3 — opzione A (padding a 768).** Motivi: (1) i kernel restano
gli stessi dot-row di gate/up e di ds4; (2) il costo è 2.5% del totale;
(3) in Q2_K il blocco è organizzato in 16 sotto-blocchi da 16 pesi
(`quants.c`, layout Q2_K): i 64 zeri di coda riempiono esattamente 4
sotto-blocchi, che ottengono scala 0 senza inquinare le scale dei pesi reali;
(4) l'imatrix assegna importanza 0 alle 64 colonne di padding. Il padding si
applica **solo** al down (Q2_K): gate/up (IQ2_XXS, che usa una griglia a
codebook dove uno zero esatto non è garantito) non ne hanno bisogno.
L'opzione B resta documentata come fallback se la validazione numerica
mostrasse artefatti dal blocco misto.

---

## 3. Ricetta proposta (variante primaria "q2-imatrix")

| Famiglia | Tensori HF (`model.language_model.*`) | Tensori GGUF (proposta §6) | Formato | bpw | Dimensione |
|---|---|---|---|---:|---:|
| Esperti routed gate+up | `layers.N.experts.gate_up_proj` (fuso, splittato in conversione) | `blk.N.ffn_gate_exps.weight`, `blk.N.ffn_up_exps.weight` | **IQ2_XXS** (+imatrix) | 2.0625 | 3.656 GiB |
| Esperti routed down | `layers.N.experts.down_proj` | `blk.N.ffn_down_exps.weight` (pad 768) | **Q2_K** (+imatrix) | 2.625 | 2.538 GiB |
| Ramo denso per layer | `layers.N.mlp.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}_shexp.weight` | Q8_0 | 8.5 | 0.530 GiB |
| Attenzione | `layers.N.self_attn.{q,k,v,o}_proj.weight` | `blk.N.attn_{q,k,v,output}.weight` | Q8_0 | 8.5 | 1.099 GiB |
| Embedding (tied, fa da LM head) | `embed_tokens.weight` | `token_embd.weight` | Q8_0 | 8.5 | 0.730 GiB |
| Router | `layers.N.router.proj.weight` | `blk.N.ffn_gate_inp.weight` | F32 | 32 | 41.3 MiB |
| Scale e norm | `router.scale`, `router.per_expert_scale`, `layer_scalar`, 7 norm/blocco, `q_norm`, `k_norm`, `norm` | vari | F32 | 32 | ~3 MiB |
| **Totale** | | | | **2.93 medi** | **≈ 8.60 GiB (9.23 GB)** |

Compressione: 48.07 GiB → 8.60 GiB ≈ **5.6×** (5.5× sul solo testo).
Sui soli esperti routed la media è 2.33 bpw.

Conti esatti (parametri → byte):

- gate+up: 15'225'323'520 par → 59'473'920 blocchi × 66 B = 3'925'278'720 B
  (blocco IQ2_XXS: 256 pesi in 66 byte, `quants.c:54`).
- down (pad 768): 8'304'721'920 par → 32'440'320 blocchi × 84 B =
  2'724'986'880 B (blocco Q2_K: 256 pesi in 84 byte, `quants.c:48`);
  senza padding sarebbero 2'497'904'640 B.
- attenzione: 1'110'179'840 par × 34/32 = 1'179'566'080 B (Q8_0: 32 pesi in
  34 byte, `quants.c:46`).
- ramo denso: 535'265'280 par × 34/32 = 568'719'360 B.
- embedding: 738'197'504 par × 34/32 = 784'334'848 B.
- router: 10'813'440 par × 4 = 43'253'760 B.

---

## 4. Motivazione di ogni scelta

### 4.1 gate/up routed a IQ2_XXS (2.0625 bpw)

Sono da soli il 60% del modello (15.2B). Scelta identica a ds4 per gli stessi
motivi: (1) il loro output passa per la non-linearità GeGLU
(`gelu(x1)*x2`, `_moe.py:346`) che attenua e satura parte dell'errore prima
che raggiunga il residual stream; (2) è la famiglia dove il risparmio paga di
più; (3) l'esperienza ds4 mostra che a questa precisione, **con imatrix**, il
modello resta utilizzabile anche in agenti di coding (`README.md:98-103`).
IQ2_XXS richiede l'imatrix (`ds4q_requires_imatrix`, `quants.c:54`): piano in
§5.

### 4.2 down routed a Q2_K (2.625 bpw, mezzo bit in più)

Asimmetria ereditata da ds4 (down = `--routed-w2 q2_k`,
`gguf-tools/README.md:94`), qui rafforzata da due argomenti specifici di
Gemma 4:

1. l'output del down entra **direttamente nel residual stream** (dopo il solo
   `per_expert_scale` e le post-norm, `_moe.py:352-364`,
   `_modules.py:682-693`): non c'è non-linearità a valle che mascheri
   l'errore, e l'errore si accumula lungo 30 layer;
2. la contrazione del down è corta (704 termini contro 2816): meno
   cancellazione statistica degli errori di quantizzazione per prodotto
   scalare, quindi varianza relativa più alta a parità di bpw.

### 4.3 Ramo denso (`mlp2`) a Q8_0

È l'equivalente funzionale dello shared expert di DeepSeek (Fase 1, §4.2) e
ds4 tiene gli shared a Q8_0 (`SExpQ8` nel nome dei GGUF, `README.md:240`).
È attraversato da **ogni token** — un suo errore è sistematico, non
diluito dal routing — e costa appena 0.53 GiB. Non vale il rischio di
scendere sotto 8.5 bpw.

### 4.4 Attenzione a Q8_0

Come ds4 (`AProjQ8`). Le proiezioni Q/K/V/O determinano il pattern di
attenzione; errori qui degradano in modo composto su tutto il contesto. Le
QK-norm di Gemma 4 (`_modules.py:236-237`) danno robustezza addizionale, ma
1.1 GiB non giustifica formati più aggressivi. Nota: sui 5 layer globali
`k_proj` fa anche da `v_proj` (K=V) — un motivo in più per non degradarla.

### 4.5 Embedding tied a Q8_0

`embed_tokens` fa sia da lookup sia da LM head (`tie_word_embeddings=true`,
`config.json:91`). ds4 tiene l'output head a Q8_0 (`OutQ8`), e qui head ed
embedding sono lo stesso tensore: Q8_0 (errore relativo ~0.3%) è ampiamente
sotto la soglia percepibile sui logits, ulteriormente compressi dal softcap
tanh a 30 (`config.json:28`). **Opzione di riserva**: bf16 (+0.65 GiB) se la
validazione dei logits (Fase 4) mostrasse sensibilità della testa.

### 4.6 Catena di routing e scale a F32 (intoccabile)

`router.proj`, `router.scale`, `per_expert_scale`, tutte le RMSNorm,
`layer_scalar`. Due motivi:

1. un errore nel router non è rumore smooth: **cambia quali esperti vengono
   selezionati** (errore discreto e potenzialmente catastrofico su token
   border-line). ds4 lascia il routing "untouched" (`README.md:102-103`);
2. `per_expert_scale` moltiplica l'output già quantizzato dell'esperto
   (`_moe.py:357-364`): quantizzarlo comporrebbe due errori moltiplicativi.

Costo totale: ~46 MiB. Irrilevante.

---

## 5. Piano imatrix (bootstrap in 3 passi, come ds4)

ds4 ha risolto lo stesso problema uovo-gallina (serve il motore per
raccogliere l'imatrix, serve l'imatrix per il q2 di qualità) così
(`gguf-tools/README.md:27-57`): prima un GGUF senza imatrix, poi raccolta con
il runtime, poi il q2 finale.

1. **Build di bring-up**: routed a **Q4_K** (non richiede imatrix,
   `quants.c:50`) e resto come da ricetta → ~14.7 GiB (§7). Serve a validare
   il motore contro i logits di riferimento prima di introdurre la variabile
   imatrix.
2. **Raccolta imatrix col motore stesso** su corpus di calibrazione
   (adattamento di `gguf-tools/imatrix/`): per ogni tensore routed, somma dei
   quadrati delle attivazioni **per colonna di input e per esperto**
   (vettori di dim 2816 per gate/up, 704 per il down — le 64 colonne di
   padding hanno importanza 0 per costruzione). Stesso formato `.dat`
   legacy llama.cpp emesso da `ds4 --imatrix-out` per riusare la pipeline.
3. **Build finale q2-imatrix**: IQ2_XXS + Q2_K con l'imatrix del passo 2.

Fallback (solo se serve un q2 prima del passo 2): euristica weight-energy
sintetica `importance[col] = Σ_righe w[col]²` già implementata dal
quantizzatore ds4 quando manca l'imatrix (`gguf-tools/README.md:111-124`).

Nota sul corpus: il dataset di calibrazione di ds4
(`gguf-tools/imatrix/dataset/`) è renderizzato col template DeepSeek; per
Gemma 4 va ri-renderizzato col template Gemma (turni `<|turn>`, §7 di Fase 1)
per attivare gli esperti su distribuzioni realistiche di prompt.

---

## 6. Contenitore: GGUF con chiavi `gemma4.*`

Come ds4, il file di distribuzione è un GGUF **specifico del motore** (non
pensato per runner generici), con naming dei tensori in stile ds4/llama.cpp:

| GGUF | Origine HF | Shape logica (post-conversione) |
|---|---|---|
| `token_embd.weight` | `embed_tokens.weight` | [262144, 2816] |
| `output_norm.weight` | `norm.weight` | [2816] |
| `blk.N.attn_q.weight` | `self_attn.q_proj.weight` | [4096, 2816] locali / [8192, 2816] globali |
| `blk.N.attn_k.weight` | `self_attn.k_proj.weight` | [2048, 2816] locali / [1024, 2816] globali |
| `blk.N.attn_v.weight` | `self_attn.v_proj.weight` | [2048, 2816], **solo 25 layer locali** |
| `blk.N.attn_output.weight` | `self_attn.o_proj.weight` | [2816, 4096] / [2816, 8192] |
| `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight` | `q_norm`, `k_norm` | [256] / [512] |
| `blk.N.ffn_gate_exps.weight` | `experts.gate_up_proj[:, 0]` | [128, 704, 2816] |
| `blk.N.ffn_up_exps.weight` | `experts.gate_up_proj[:, 1]` | [128, 704, 2816] |
| `blk.N.ffn_down_exps.weight` | `experts.down_proj` + pad | [128, 2816, 768] |
| `blk.N.ffn_gate_inp.weight` | `router.proj.weight` | [128, 2816] |
| `blk.N.router_scale` / `blk.N.per_expert_scale` / `blk.N.skip_scale` | `router.scale` / `router.per_expert_scale` / `layer_scalar` | [2816] / [128] / [1] |
| `blk.N.ffn_{gate,up,down}_shexp.weight` | `mlp.{gate,up,down}_proj.weight` | [2112, 2816] ×2, [2816, 2112] |
| le 7 norm di blocco | `*_layernorm*` | [2816] |

Lo split del tensore fuso usa l'asse k di `gating_einsum`: k=0 → gate (riceve
la GELU), k=1 → up (`_moe.py:343-346`). Il riuso del suffisso `_shexp` per il
ramo denso è deliberato: il motore lo tratta esattamente come ds4 tratta lo
shared expert (sempre attivo, alta precisione).

Chiavi di metadata proposte (validate dal motore all'avvio, come fa ds4 con
`deepseek4.*`, `ds4.c:2028-2030`):

```text
gemma4.block_count                 = 30
gemma4.context_length              = 262144
gemma4.embedding_length            = 2816
gemma4.attention.head_count        = 16
gemma4.attention.head_count_kv     = 8
gemma4.attention.global_head_count_kv = 2
gemma4.attention.key_length        = 256
gemma4.attention.global_key_length = 512
gemma4.attention.k_eq_v_global     = true
gemma4.attention.sliding_window    = 1024
gemma4.attention.layer_pattern     = "LLLLLG"
gemma4.attention.layer_norm_rms_epsilon = 1e-6
gemma4.rope.local.freq_base        = 10000.0
gemma4.rope.global.freq_base       = 1000000.0
gemma4.rope.global.partial_factor  = 0.25
gemma4.expert_count                = 128
gemma4.expert_used_count           = 8
gemma4.expert_ffn_length           = 704
gemma4.expert_down_padded_length   = 768
gemma4.dense_ffn_length            = 2112
gemma4.final_logit_softcap         = 30.0
gemma4.activation                  = "gelu_tanh"
gemma4.tie_embeddings              = true
gemma4.eos_token_ids               = [1, 106]
```

---

## 7. Varianti previste (stile ds4)

| Variante | Routed | Dimensione stimata | Classe macchina (pesi + KV + scratch) |
|---|---|---:|---|
| **q2-imatrix** (primaria) | gate/up IQ2_XXS, down Q2_K | **8.60 GiB** | 16 GB RAM |
| q4-imatrix (bring-up e qualità) | tutti Q4_K | 14.73 GiB | 24-32 GB RAM |
| q2-q4-imatrix (mista) | ultimi 6 layer Q4_K, resto q2 | 9.82 GiB | 16 GB RAM |
| q2-down-q4 (leva qualità) | gate/up IQ2_XXS, down Q4_K | 10.41 GiB | 16 GB RAM |

Note:

- la variante mista replica il knob ds4 `q2-q4-imatrix` ("last 6 layers q4",
  `README.md:109`): i layer finali incidono di più sulla forma dei logits;
- il rapporto pesi/RAM cambia radicalmente la fascia target rispetto a ds4
  (81-91 GiB → 8.6 GiB): l'obiettivo "gira su macchine comuni" è realistico
  anche senza SSD streaming, che resta utile sotto i 12-16 GB di RAM;
- granularità streaming: un esperto quantizzato q2 pesa **1.65 MiB**
  (gate+up 1'022'208 B + down 709'632 B), circa 4× più fine dell'esperto
  Flash (~6.8 MiB): cache più efficiente e miss meno costosi, worst case
  cold ≈ 396 MiB/token (30 layer × 8 esperti).

---

## 8. Impatto qualità atteso e rischi

Aspettative (qualitative — i numeri arriveranno dalla validazione di Fase 4):

- **Precedente ds4**: 2-bit asimmetrico con imatrix su Flash "si comporta
  bene, funziona sotto coding agent e chiama i tool in modo affidabile"
  (`README.md:98-103`), con l'85% circa della massa a ~2.3 bpw.
- **Frazione attiva quantizzata a 2 bit più bassa che in Flash**: per token,
  su ~3.8B parametri attivi solo ~1.43B (37%) stanno negli esperti routed
  2-bit; attenzione, ramo denso, router ed embedding/head (63%) restano a
  ≥8.5 bpw. In Flash la quota routed del compute attivo è più alta (~50%).
  A parità di bpw sui routed, l'errore per step dovrebbe pesare meno.
- La ridondanza del MoE (128 esperti, top-8 con pesi rinormalizzati) e
  l'imatrix per-esperto sono le difese principali.

Rischi specifici e mitigazioni:

| Rischio | Perché | Mitigazione |
|---|---|---|
| `expert_dim=704` piccolo | meno ridondanza interna per esperto rispetto ai 2048 di Flash; ogni peso "conta" di più | imatrix reale (non sintetica) per il build finale; variante q2-down-q4 |
| Blocco Q2_K misto (192 reali + 64 pad) | scale di super-blocco calcolate su blocco parzialmente vuoto | i 64 zeri riempiono esattamente 4 sotto-blocchi da 16 → scala 0 isolata; verifica con `--compare-tensor` in conversione |
| top-k approssimato in training (`approx_max_k`, `_moe.py:304`) vs top-8 esatto nel motore | routing leggermente diverso dall'implementazione ufficiale su logit quasi pari | confronto logits ufficiali (metodologia `tests/test-vectors`); il config HF non richiede l'approssimazione |
| `per_expert_scale` amplifica il rumore | moltiplica output già quantizzati | tenuto F32; monitorare i layer con scale estreme in validazione |
| Testa tied quantizzata | lookup e head condividono l'errore Q8_0 | opzione bf16 per `token_embd` (+0.65 GiB) |

Criterio di accettazione (da formalizzare in Fase 4, mutuando
`gguf-tools/quality-testing/`): confronto a campione con le continuazioni
ufficiali e delta di score tra build q4 (bring-up) e q2 finale entro la
stessa banda osservata da ds4 tra i suoi q4/q2.

---

## 9. Decisioni riassunte

1. **P1**: motore text-only, vision tower esclusa.
2. **P2**: quantizzazione aggressiva solo sui routed (90.6% della massa).
3. **P3**: down con zero-padding 704→768 (kernel invariati; +2.5% totale).
4. Ricetta primaria: **IQ2_XXS (gate/up) + Q2_K (down) + Q8_0 (denso,
   attenzione, embedding) + F32 (router/scale/norm)** → **≈8.60 GiB**,
   2.93 bpw medi.
5. Bootstrap: Q4_K → raccolta imatrix col motore → build q2-imatrix finale.
6. Contenitore GGUF con chiavi `gemma4.*`, split gate/up in conversione,
   convenzione `_shexp` riusata per il ramo denso.
