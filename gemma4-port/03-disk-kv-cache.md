# Fase 3 — Disk KV cache per Gemma 4 26B-A4B (formato G4KV/G4SP)

Progettazione del KV cache su SSD come cittadino di prima classe, derivata
dal design ds4 (sezione "Disk KV Cache" di `README.md:990-1172`, header
`ds4_kvstore.h`, implementazione `ds4_kvstore.c`, payload `DSV4` in
`ds4.h:302-333`) e adattata all'attenzione ibrida sliding/global di Gemma 4
accertata in Fase 1.

---

## 1. Sintesi del design ds4 (cosa si eredita)

Meccanismi di ds4 che si trasferiscono invariati, perché indipendenti
dall'architettura del modello:

1. **Chiave di lookup = SHA1 del prefisso di byte renderizzato**, non dei
   token: il file `<sha1>.kv` è riusabile sse i byte cached sono un prefisso
   del prompt renderizzato in arrivo; i token esatti restano autoritativi nel
   payload e solo il suffisso testuale nuovo viene tokenizzato
   (`README.md:1011-1016, 1055-1060`). Questo assorbe i mismatch di
   ritokenizzazione BPE/SP tra ciò che il modello ha generato e ciò che il
   client rimanda.
2. **Contenitore a sezioni**: header fisso 48 byte + testo renderizzato +
   payload di stato + sezioni trailer opzionali a bit di estensione (tool-id
   map "KTM", titolo sessione, ecc., `README.md:1027-1096`,
   `ds4_kvstore.h:15-18`, hook generici `ds4_kvstore_trailer_hooks` in
   `ds4_kvstore.h:92-99`).
3. **Momenti di salvataggio**: `cold` / `continued` / `evict` / `shutdown`,
   con trim del suffisso fragile e allineamento a frontiere assolute
   (`README.md:1139-1163`).
4. **Eviction a densità di valore**: score =
   `(hit_effettivi + 1) × token / byte_file`, con decadimento esponenziale
   degli hit (half-life 6h, `ds4_kvstore.h:14`), bonus per i checkpoint
   "anchor" (cold/evict/shutdown) e penalità per i `continued` superati da un
   checkpoint in arrivo più lungo sullo stesso prefisso
   (`ds4_kvstore.c:504-560`).
5. **Scrittura sicura**: staging su file temporaneo e scrittura con
   `read`/`write` ordinari, niente `mmap` (`README.md:1017-1019`,
   `ds4_session_stage_payload`, `ds4.h:311-317`).
6. **Logits del token successivo salvati nel payload** (`f32[vocab]`), così
   un resume riparte campionando subito senza un decode extra
   (`README.md:1122-1127`).
7. **Tool-id map**: `tool id → blocco tool-call campionato esatto`, per
   ri-renderizzare la history del client byte-per-byte (`README.md:1071-1096`).
   Per Gemma 4 il blocco memorizzato è
   `<|tool_call>call:NOME{...}<tool_call|>` invece del DSML.

Ciò che invece **non** si trasferisce è il contenuto per layer del payload
`DSV4` (righe raw + righe compresse + frontiere del compressor + stream
indexer, `README.md:1108-1121`): lo stato di Gemma 4 è puramente per-token,
senza frontiere di compressione.

---

## 2. Perché per Gemma 4 il disk-KV è ancora più centrale: il rewind

Analisi che motiva l'intero progetto. Con l'attenzione sliding-window, lo
stato in RAM dei 25 layer locali è un ring degli **ultimi 1024** token
(`_modules.py:301-317`; maschera `_modules.py:38-52`).

Conseguenza: **il rewind in RAM è impossibile** oltre il token corrente.
Per riportare la sessione dalla posizione `T` a una posizione `p < T` servono
le righe K/V locali delle posizioni `p-1023..p-1`; il ring ne contiene solo
`T-1024..T-1`. Ricalcolarle richiederebbe le attivazioni del layer
precedente su quelle posizioni, che a loro volta richiedono le finestre
locali di posizioni ancora precedenti: la ricorsione termina solo a 0 o a un
checkpoint salvato. (I layer globali invece si troncano banalmente: le righe
`≤ p` restano valide.)

In ds4 il problema è attenuato dal design: la finestra raw è di soli 128
token e la storia compressa è ricostruibile in parte; esiste comunque il
fallback "restore da snapshot su disco + replay del suffisso"
(`README.md:788-792`, `ds4.h:238-254`). In Gemma 4 quel fallback è
**l'unico** meccanismo corretto di rewind: editing dell'ultimo messaggio,
branching della conversazione, ritocchi della history da parte di client
agentici — tutti casi comuni con le API stateless — passano necessariamente
da un checkpoint su disco.

Da qui la regola di progetto: **la cadenza dei salvataggi `continued` limita
il costo massimo di un rewind** (replay ≤ intervallo + allineamento). Il KV
su disco non è un'ottimizzazione: è il meccanismo di sessione.

---

## 3. Cosa si serializza per layer

| Layer | Stato vivo in RAM | Stato serializzato |
|---|---|---|
| Locale (×25) | ring K/V `[1024, 8, 256]` bf16 + posizioni | `R = min(T, 1024)` righe K poi V, **in ordine logico** (posizioni `T-R..T-1`), 8 KiB/riga totali |
| Globale (×5) | K/V `[T, 2, 512]` bf16 | `T` righe K poi `T` righe V, 4 KiB/token |
| Comune | posizioni, end_index | **non serializzati**: motore text-only batch 1, posizioni = `0..T-1` ricostruite; una futura variante multimodale (token immagine inseriti) richiederà bump di versione |

L'ordine logico delle righe locali (non l'ordine fisico del ring) replica la
scelta ds4 (`README.md:1112-1114`): il loader ricostruisce il ring con
qualunque capacità di contesto, e un checkpoint scritto con `ctx` piccolo si
carica in una sessione con `ctx` più grande (stessa proprietà sfruttata
dall'eviction ds4, `ds4_kvstore.c:517-519`).

**Decisione D1 — sui layer globali si salvano sia K sia V,** anche se
derivano dalla stessa proiezione (K=V pre-norm). Alternativa valutata e
scartata: salvare la proiezione pre-norm una sola volta (−50% dello stato
unbounded) e riapplicare `key_norm`+RoPE / `value_norm` al load. Scartata
perché il motore in RAM tiene K e V già post-processati (come il grafo JAX,
`_modules.py:277-317`): la proiezione pre-norm non esiste più al momento del
save, e le RMSNorm non sono invertibili (il fattore rms per riga è perso).
Tenerla in RAM solo per i save costerebbe +2 KiB/token/layer residenti,
esattamente ciò che si voleva risparmiare su disco.

**Decisione D2 — dtype delle righe: bf16 verbatim** (memcpy in save e load,
zero ricomputo). Estensione futura (flag nell'header): righe globali Q8_0 su
disco (~−47%), da valutare solo dopo la validazione numerica del motore.

---

## 4. Formato del file: contenitore `G4KV`

Layout identico al KVC di ds4 (`README.md:1029-1053`) con magic e model-id
propri; little-endian.

```text
0   u8[3]  magic = "KVG"
3   u8     version = 1
4   u8     bit di quantizzazione degli esperti routed (2 o 4)
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags: bit 0 = tool-id map, bit 1 = titolo sessione
7   u8     model_id: 1 = Gemma 4 26B-A4B (0 riservato/invalido)
8   u32    token count del checkpoint
12  u32    hit count
16  u32    context size di scrittura
20  u8[4]  riservati
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    byte del payload G4SP
```

Seguono, come in ds4:

```text
u32                  rendered_text_bytes
u8[...]              testo renderizzato del prefisso (identità del file: SHA1 → nome)
u8[payload_bytes]    payload G4SP (sezione 5)
[sezioni trailer]    tool-id map "KTM" (formato identico a ds4, README.md:1071-1082)
```

Il campo `model_id` segue la logica di `ds4_engine_model_id`
(`ds4.h:157-160`, `ds4_kvstore.h:43-45`): identifica la *shape*, così cache
di modelli futuri non vengono mai caricate per errore. I `quant_bits` in
header replicano la politica ds4 sul riuso tra varianti q2/q4
(`README.md:1166-1169`): default permissivo — lo stato K/V è prodotto dal
path d'attenzione Q8_0, identico tra le varianti; le differenze indotte dagli
esperti sulle hidden state sono accettate come in ds4 — con flag di
strictness `--kv-cache-reject-different-quant`.

---

## 5. Payload `G4SP` (Gemma 4 Session Payload)

Header di sedici `u32` little-endian (mutuato dai tredici del `DSV4`,
`ds4.h:304-309`, `README.md:1091-1107`; i campi shape validano il checkpoint
contro il modello caricato, come fa ds4 con layer count e dimensioni):

```text
0   magic = "G4SP"
1   payload version = 1
2   context size del salvataggio
3   prefill chunk size
4   layer count            = 30
5   pattern period         = 6   (layer globale a indice 5 di ogni periodo)
6   sliding window         = 1024
7   n_kv_heads locali      = 8
8   key dim locale         = 256
9   n_kv_heads globali     = 2
10  key dim globale        = 512
11  vocab size             = 262144
12  token count T
13  kv dtype               = 0 (bf16)
14  flags                  bit 0 = logits presenti
15  riservato
```

Corpo, nell'ordine:

```text
u32[T]           token id del checkpoint
f32[262144]      logits del token successivo (1.0 MiB, come DSV4)
per ogni layer locale (indici 0,1,2,3,4, 6,7,... in ordine):
    u32          R = min(T, 1024)
    bf16[R][8][256]   righe K, posizioni T-R..T-1 in ordine logico
    bf16[R][8][256]   righe V
per ogni layer globale (indici 5,11,17,23,29 in ordine):
    u32          R = T
    bf16[T][2][512]   righe K
    bf16[T][2][512]   righe V
```

---

## 6. Dimensioni e budget

Costanti: righe locali = 200 MiB fissi per T ≥ 1024 (25 layer × 8 MiB);
righe globali = 20 KiB/token; logits 1 MiB; token 4 B/token.

| T (token) | Globali | Totale file | Scrittura @2 GB/s | Re-prefill @500 t/s |
|---:|---:|---:|---:|---:|
| 8'192 | 160 MiB | ≈ 361 MiB | 0.18 s | 16 s |
| 32'768 | 640 MiB | ≈ 841 MiB | 0.42 s | 66 s |
| 131'072 | 2.50 GiB | ≈ 2.70 GiB | 1.4 s | 262 s |
| 262'144 | 5.00 GiB | ≈ 5.32 GiB | 2.7 s | 524 s |

Il rapporto costo-scrittura / costo-ricomputo è 100-200×: la tesi ds4 ("la KV
cache appartiene al disco") regge su Gemma 4 con margine. Nota comparativa: a
parità di token un checkpoint Gemma 4 pesa come uno ds4 Flash (~26 KiB/token,
`README.md:829-832` vs ~20.8 KiB/token qui), ma il *modello* è 10× più
piccolo: su una macchina da 16 GB il budget disco per la cache KV
(`--kv-disk-space-mb`, default 4 GiB come `DS4_KVSTORE_DEFAULT_MB`,
`ds4_kvstore.h:13`) sarà tipicamente più grande dei pesi stessi.

Peculiarità di Gemma 4 sfruttabile in futuro (v2): le righe globali sono
**append-only** (le K/V di posizioni passate non cambiano mai), quindi un
save `continued` potrebbe scrivere solo il delta globale dall'ultima
frontiera + la finestra locale corrente + i logits (~200 MiB + delta invece
del file intero). V1 resta a file autocontenuti come ds4, per semplicità di
eviction e crash-safety.

---

## 7. Frontiere di salvataggio

Politica e default ereditati 1:1 da ds4 (`README.md:1139-1163`, opzioni
`ds4_kvstore_options`, `ds4_kvstore.h:59-66`):

| Momento | Trigger | Note |
|---|---|---|
| `cold` | primo prompt lungo stabile, prima della generazione | trim 32 token di coda, align-down a 2048; min 512, max 30'000 token |
| `continued` | il grafo vivo attraversa una frontiera assoluta allineata (~ogni 10'000 token) | indipendente da dove è caduto il cold |
| `evict` | una richiesta non correlata sta per sostituire la sessione viva | l'unico modo di riprendere senza re-prefill (una sola sessione viva in RAM, `README.md:999-1003`) |
| `shutdown` | uscita pulita | |

Regola di cattura (stessa semantica ds4): lo stato si cattura **nel momento
in cui la posizione viva attraversa la frontiera** — durante il prefill
chunked (4096 default) o la generazione — con staging su file temporaneo e
rename atomico; mai ricostruito retroattivamente. Per Gemma 4 questo è anche
un vincolo di correttezza, non solo di design: per la sezione 2, la finestra
locale alla frontiera `F` esiste solo mentre la posizione viva è `F` (con il
ring da 1024 righe non si può "tornare indietro" di più del margine
trim+align, 32+2048 > 1024).

Trade-off esplicito del knob `continued-interval` (default 10'000): è anche
il **limite superiore del replay in caso di rewind** (§2). Sessioni agentiche
con molte modifiche della history possono abbassarlo a 4'096-8'192 pagando
più scritture (0.2-0.8 s l'una, asincrone).

---

## 8. Hit, miss e riuso dei prefissi

Pipeline di matching identica a ds4 (`README.md:990-1016`):

1. **Sessione viva**: confronto esatto del prefisso di token
   (`ds4_session_common_prefix`); se il prompt in arrivo estende il
   checkpoint vivo si valuta solo il suffisso.
2. **Miss sul vivo → lookup su disco**: si renderizza il prompt in byte, si
   cercano le entry il cui testo è prefisso dei byte in arrivo (directory
   scandita e tenuta in RAM, refresh come `kv_cache_refresh`,
   `ds4_kvstore.c:468-483`), si preferisce il prefisso più lungo compatibile
   (`model_id`, `quant_bits` se strict, `ctx_size` ≤ ctx corrente).
3. **Load**: restore del payload → il ring locale si ripopola dalle righe
   logiche, i K/V globali si copiano, i logits salvati permettono di
   campionare subito; si tokenizza solo il suffisso testuale nuovo e si
   prefilla solo quello.
4. **Rewind/edit della history** (prompt in arrivo *non* estende il vivo):
   niente riscrittura in place (impossibile per §2) — si cerca su disco il
   miglior checkpoint che sia prefisso del nuovo prompt e si replaya il
   suffisso; in assenza, re-prefill da zero.
5. **Touch degli hit**: contatore e last-used aggiornati in-place nel header
   (`ds4_kvstore_touch_file`, `ds4_kvstore.c:485`).

Eviction: formula ds4 portata pari pari (`ds4_kvstore_entry_eviction_score`,
`ds4_kvstore.c:532-560`):

```text
score = (hit_effettivi + 1) × token / byte_file
hit_effettivi = hit × 2^(−età_ultimo_uso / 6h)
× fattore bonus se reason ∈ {cold, evict, shutdown}   (anchor)
× fattore penalità se "continued" superato da un checkpoint in arrivo
  più lungo sullo stesso prefisso (stessa SHA1 del sotto-prefisso,
  ds4_kvstore.c:504-527)
```

La penalità di supersede è particolarmente adatta a Gemma 4: dato il §6, i
`continued` intermedi di una stessa conversazione condividono i 200 MiB di
finestra locale e differiscono solo per le righe globali — tenere solo il più
lungo è quasi sempre corretto.

---

## 9. Cosa deliberatamente NON si fa: KV streaming a compute-time

Per onestà di progetto: "KV su SSD" in ds4 (e qui) significa persistenza e
resume, **non** leggere la KV dall'SSD durante l'attenzione. Per Gemma 4
l'idea è stata valutata e scartata:

- ogni step di decode legge *tutte* le righe globali (attention densa sulla
  storia): a 128k di contesto sarebbero 2.5 GiB di letture per token —
  impraticabile senza un meccanismo di sparsità alla DeepSeek (indexer
  top-k, `MODEL_CARD.md:45-48`) che Gemma 4 non ha;
- non serve: il design del modello fa già il lavoro — lo stato totale a
  contesto **massimo** (256k) è ~5.3 GiB, che sta in RAM anche su macchine
  da 16 GB accanto agli 8.6 GiB di pesi q2 (Fase 2 §7).

L'analogo Gemma-side dell'SSD streaming di ds4 resta quello degli **esperti**
(Fase 2 §7, granularità 1.65 MiB), non della KV.

---

## 10. Riepilogo decisioni

1. **D1**: sui layer globali si serializzano K e V entrambi (bf16 verbatim);
   niente ricostruzione dalla proiezione condivisa.
2. **D2**: dtype su disco = bf16; Q8_0 per riga come flag futuro.
3. Contenitore `KVG` v1 = layout KVC a 48 byte con `model_id=1`; chiave =
   SHA1 del prefisso renderizzato; trailer KTM per l'exact-replay dei tool
   call `<|tool_call>call:...`.
4. Payload `G4SP` v1: 16 campi u32 di shape-check, token, logits f32,
   finestre locali in ordine logico (bounded, 200 MiB), righe globali
   complete (20 KiB/token).
5. Posizioni non serializzate (ricostruite `0..T-1`); versione da bumpare per
   il multimodale.
6. Frontiere, trim/align, staging e eviction score: politica ds4 invariata;
   il `continued-interval` è documentato come limite del replay di rewind.
7. Il rewind è *solo* disk-checkpoint + replay: con SWA non esiste rewrite in
   place (analisi §2) — è l'argomento più forte a favore del porting
   dell'idea ds4 su questa architettura.
