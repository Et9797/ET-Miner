# Senior-Reviewer Pass — ET-Miner Paper (Engels)

**Datum:** 2026-06-01
**Doelbestand:** `/home/et/personal-projects/et-miner/papers/et_miner_proteome.tex` (968 regels)
**Scope:** drie assen — (1) fout geciteerde referenties, (2) numerieke consistentie,
(3) spelling & grammatica. Engels-only. Geen wijzigingen aangebracht aan de paper; dit is
**alleen een review-rapport**.
**Methode:** drie parallelle senior-reviewer subagents (referentie-audit mét volledige
web-verificatie · interne getal-consistentie · taal/typografie), gevolgd door handmatige
verificatie van de zwaarste bevindingen tegen de `.tex` (regelnummers + quotes geverifieerd;
kdist-som handmatig nageteld; orphan-refs via `grep \cite` bevestigd).

---

## Verdict in één alinea

De **rekenkundige kern is opvallend gezond** — de twee grote tabel-sommen kloppen tot op de
eenheid (kdist K=1..22 = 26,849,505 exact; expanded cumulatief = 16,812,646,639 exact), en
álle min_counts, groeifactoren, Z-scores en speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×)
kloppen binnen afronding. Et's vermoeden "nummers die allemaal kloppen" klopt grotendeels.
**De echte problemen zitten elders:** in de **referenties** (meerdere foute auteurs/pagina's,
één claim die een concurrent feitelijk verkeerd weergeeft), in de **geheugen-cijfers**
(dezelfde data krijgt 3 verschillende CSR-groottes via 3 byte-conventies, zonder labels), en
één **grammaticaal kapotte zin in de abstract**. Niets ondermijnt de hoofdresultaten, maar
de citatie-fouten zijn een integriteitskwestie die een editor zal opmerken.

---

## AS 1 — Foute / problematische referenties

> Bibliografie: 37 `\bibitem` (r574–757). 32 distincte `\cite`-keys, allemaal resolvend
> (geen broken refs). Elke fout hieronder is web-geverifieerd met bron-URL.

### MAJOR — citatie-integriteit

| # | Ref | Regel | Probleem | Correctie | Bron |
|---|-----|-------|----------|-----------|------|
| R1 | `barrio2024` | 660 | **Verkeerde eerste auteur.** Staat als "I.~Barrio-Hernandez et al."; werkelijke eerste auteur is **Andy M. Lau** (Lau, Bordin, Kandathil, Sillitoe, Waman, Wells, Orengo, Jones). Barrio-Hernandez staat niet op de auteurslijst — waarschijnlijk verwisseld met Barrio-Hernandez 2023 (AlphaFold-clusters, Nature). Venue/vol/artikel kloppen wél. | `A.~M.~Lau et al.` | [DOI 10.1126/science.adq4946](https://pubmed.ncbi.nlm.nih.gov/39480926/) |
| R2 | `coin2009` | 590 | **Verkeerde eerste auteur (en cite-key naam misleidend).** Staat als "L.~Coin et al."; werkelijke auteurs zijn **N. Terrapon, O. Gascuel, E. Maréchal, L. Bréhélin**. Pagina's ook incompleet (3077 → 3077–3083). Ge-cite'd op r102 en r499 als "Coin et al." | `N.~Terrapon et al.` + pagina's `3077--3083` (cite-key mag blijven, maar "Coin et al." in lopende tekst r499 → "Terrapon et al.") | [DOI 10.1093/bioinformatics/btp560](https://academic.oup.com/bioinformatics/article/25/23/3077/216103) |
| R3 | `varadi2022` | 582 | **Verkeerde pagina-range.** `50(D1):D419--D427` → moet **D439–D444** zijn. Bevestigd: de foutieve range `D419` collideert exact met de echte range van `mistry2021` (`D412--D419`, r712) → klassieke copy-fout. | `50(D1):D439--D444` | [DOI 10.1093/nar/gkab1061](https://academic.oup.com/nar/article/50/D1/D439/6430488) |
| R4 | `chon2018b` (BIGMiner) | claim r452, r467 | **Claim geeft concurrent feitelijk verkeerd weer — en ondermijnt jullie eigen framing.** Tabel zegt BIGMiner = "100M transactions / 30× servers". De BIGMiner-paper rapporteert schaling tot **6,5 miljard** transacties (≈65× meer dan de geclaimde 100M). Het "30 nodes"-getal is niet verifieerbaar uit open bronnen. Door BIGMiner op "100M" te zetten verzwak je juist de claim "three orders of magnitude beyond any prior result". | Corrigeer naar de werkelijke gepubliceerde schaal (6.5B, distributed MapReduce), en herformuleer de "orders of magnitude"-claim als **single-GPU** vergelijking i.p.v. absoluut. | [Springer 10.1007/s10586-018-1812-0](https://link.springer.com/article/10.1007/s10586-018-1812-0) |

### MINOR — bibliografische details

| # | Ref | Regel | Probleem | Correctie | Bron |
|---|-----|-------|----------|-----------|------|
| R5 | `meysman2015` (claim) | 499 | Claim-support: "limited to ${\sim}100$K structures" — werkelijk **~32.142** structuren (overschat ~3×). "3D-proximity / PDB" klopt wél. | `${\sim}32$K structures` | [PMC4318390](https://pmc.ncbi.nlm.nih.gov/articles/PMC4318390/) |
| R6 | `acmsurvey2021` | 637 | Issue + pagina's fout: `54(7):1--36` → **54(9), Article 179 (179:1–179:35)**. | `54(9):179:1--179:35` | [DOI 10.1145/3472289](https://dl.acm.org/doi/abs/10.1145/3472289) |
| R7 | `fang2009` | 721 | Titel/venue-mismatch: de bib-titel "Parallel data mining on graphics processors" is de titel van het **2008 HKUST tech-report**; de DaMoN-2009 workshop-paper heet "**Frequent itemset mining on graphics processors**" (pp 34–42). | Titel + pagina's corrigeren | DaMoN 2009, pp 34–42 |
| R8 | `chon2024` | 651 | Titel verkeerd geparafraseerd: "Reducing redundant computations in GPU-based frequent itemset mining" → werkelijke titel "**Boosting GPU-based frequent itemset mining by reducing redundant computations**". ESWA vol 250, Art. 123928. | Titel corrigeren | [DOI 10.1016/j.eswa.2024.123928](https://www.sciencedirect.com/science/article/abs/pii/S0957417424007942) |
| R9 | `varadi2024` | 642 | Ontbrekende eind-pagina: `52(D1):D368` → **D368–D375**. | `52(D1):D368--D375` | [DOI 10.1093/nar/gkad1011](https://academic.oup.com/nar/article/52/D1/D368/7337620) |
| R10 | `chon2018b` | 742 | Eind-pagina off-by-one: `21:1507--1521` → **21(3):1507–1520**. | `21(3):1507--1520` | Springer (zie R4) |
| R11 | `savasere1995` | 646 | Pagina's ontbreken volledig. | `pp.\ 432--444` | [VLDB 1995 P432](https://www.vldb.org/conf/1995/P432.PDF) |
| R12 | Speedup-cluster | 106 | `\cite{luna2019,han2000,zaki2000,zhang2011,chon2018,djenouri2019,acmsurvey2021}` voor "GPU achieved 50–350× speedups" bundelt twee **CPU-algoritmes** (`han2000` FP-Growth, `zaki2000` Eclat) onder een GPU-speedup-claim. De range zelf klopt (350× = djenouri2019, 100× = zhang2011, beide bevestigd). | `han2000`/`zaki2000` uit dit cluster halen | — |
| R13 | GMiner txn-count | 466 vs 877 | Zelfde systeem krijgt "15M" (scale-tabel, basis van de 5.1×-claim) én "1.7M (real)" (gpu-arch-tabel). 1.7M = webdocs FIMI (bevestigd 1,692,082); 15M is vermoedelijk synthetisch maar niet zo gelabeld → ogenschijnlijke zelf-tegenspraak. (GPU "4× GTX 1080" niet verifieerbaar — paywall.) | Label "15M (synthetic) / 1.7M (real)" | FIMI webdocs |

### HOUSEKEEPING — orphan-referenties (bevestigd 0× ge-cite'd)

`webb2007` (r724), `webb2014` (r729), `abramson2024` (r734), `zaki1997` (r749),
`miettinen2020` (r754) — allemaal bibliografisch correct, maar **nergens ge-`\cite`d**.
Ofwel citeren ofwel verwijderen. **`abramson2024` (AlphaFold 3) is de natuurlijke citatie
voor de complex-structuren-discussie (r537–539) en ontbreekt daar opvallend** — dáár inzetten
lost meteen één orphan op.

---

## AS 2 — Numerieke consistentie

> Conclusie vooraf: de wetenschappelijke hoofdgetallen kloppen. Alle FOUT/DUBIEUS-vlaggen
> betreffen **labeling, eenheidsconventies, of triviale dollar-rekensommen**.

### MAJOR / verwarrend — geheugen-cijfers (de enige echte rijm-zone)

| # | Probleem | Regel(s) | Detail |
|---|----------|----------|--------|
| N1 | **CSR krijgt drie groottes via drie byte-conventies** | r175, r179/r485, r920 | Body: "316M nnz, ~5.1 GB, **two** 64-bit ints/entry" (16 B/entry, COO-paar). Elders: H2D-transfer "~3 GB" (alleen kolom-index, 8 B → 316M×8=2.53 GB). Appendix-tabel: "CSR (**8** bytes/entry)" → 19 GB voor 214M. Drie cijfers (3 / 5.1 / 19 GB) voor "CSR" zonder dat de lezer ze kan rijmen. **Fix:** label expliciet (COO-paar 2×i64 vs CSR-kolomindices i64) en geef de nnz van de 214M-set (nu nergens vermeld). |
| N2 | **"40× reduction" vs "1.4× reduction" vergelijken stilzwijgend verschillende dingen** | r175 vs r908/r920 | 40× = naïeve **byte**-matrix (206 GB, 1 B/boolean) ÷ CSR-COO (5.1 GB). 1.4× = **bitpacked** dense (27 GB) ÷ CSR (19 GB). Beide rekenkundig correct, maar het "206 GB dense" is een representatie die je **nooit gebruikt** (de echte on-GPU bitvector is 26 GB = de "27 GB" appendix-dense, factor ~8 kleiner). **Fix:** in body verduidelijken dat 206 GB = 1 byte/boolean (niet bitpacked), zodat de 40×-claim niet als appels-met-peren leest. |
| N3 | **`~445 GB` expanded bitvector: GB/GiB-mix** | r258 | "~56 GB/device × 8" = 448, niet 445. Decimaal 109.2M×35012/8 = **478 GB**. 445 klopt **alleen** als **GiB** (445.19 GiB). Drie getallen die niet samenvallen door GB↔GiB-wissel. **Fix:** kies één conventie; als GiB, dan is "56 GB/device" ook fout (→ 55.6 GiB). |

### MAJOR — statistiek

| # | Claim | Regel | Probleem |
|---|-------|-------|----------|
| N4 | "$p < 0.17$ by one-sided binomial test" (0/5 permutaties > K=6) | 414, 429, 435 | **Niet onderbouwd door de standaard one-sided binomiale upper bound.** Voor 0 successen in 5 trials is de 95%-upper-bound ≈ **0.45** (rule-of-three 3/5 = 0.6). 0.17 ≈ 1/6 — herkomst onduidelijk en oogt te optimistisch. **Fix:** vervang door ~0.45, óf expliciteer de exacte aanname waaruit 0.17 volgt. (Sluit aan bij de eerdere review-flag over de zwakke 5-permutatie-basis.) |

### MINOR / NIT

| # | Claim | Regel | Rekensom | Fix |
|---|-------|-------|----------|-----|
| N5 | min_count expanded = 1,092 | 260 | ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093**; pseudocode r800 specificeert ⌈σ·n⌉, en Power gebruikt wél ceil (769). Richting inconsistent. | 1092→1093, óf vermeld dat round() gebruikt is en pas r800 aan. |
| N6 | "$243 saving" | 965 | 332 − 90 = **242**, niet 243. | `\$242` |
| N7 | "$332 ≈ $3.50/GPU/uur × 8 × ~12h" | 965 vs 940 | 3.50×8×12 = **$336**; $332 impliceert 11.86 h (≠ "~12 hours"). | `\$336`, of expliciteer afwijkende uren. |
| N8 | "excludes ... (62.6%)" (Limitation 4) | 525 | Getal correct voor de **base**-run (37.4+62.6=100); niet gelabeld als zodanig → spanning met de 53.1%-expanded-claim. | "(base vocabulary)" toevoegen. |
| N9 | "26.8M total" | 301, 306 | 26,849,505 → 26.85M; "26.8M" is **truncatie** i.p.v. afronding (zou 26.9M zijn). Cosmetisch. | Optioneel "26.8M" → "26.85M". |

### GEVERIFIEERD CORRECT (selectie, ter geruststelling)

kdist-som = 26,849,505 (handmatig nageteld ✓) · expanded cumulatief = 16,812,646,639 ✓ ·
alle 7 groeifactoren (25.8×…3.4×) ✓ · alle min_counts (76,891 … 8) ✓ · 37.4% / 53.1% /
13.14% ✓ · Z>3,700 voor K=4–6 consistent abstract↔tabel↔conclusie ✓ · 9× / 124× / 21× /
95.2% / 5.1× ✓ · null Bio-kolom = 475,865 = Direct-GPU-totaal ✓ · pruning 1−(0.32)(0.88)(0.96)
= 73% ✓ · ~264 B totaal transfer (22×12) ✓.

---

## AS 3 — Spelling & grammatica

> Geen echte spelfouten gevonden. Oxford-comma, `vs.\`, `et al.~` en en-dash-ranges
> (`70--90`, `583--589`, `$K{=}4$--$8$`) zijn consistent en correct toegepast.

### MAJOR

| # | Regel | Huidige tekst | Probleem | Fix |
|---|-------|---------------|----------|-----|
| G1 | 92 (abstract) | "Three novel kernel-level pruning techniques, including ... `\_\_ballot\_sync`, **we estimate would reduce** $K{=}9$ runtime ..." | **Grammaticaal kapot:** de zin opent met een naamwoordgroep ("Three novel ... techniques") die nooit een hoofdwerkwoord krijgt; "we estimate would reduce" propt midden in de zin een nieuw onderwerp ("we"). | Herschrijf zoals **r549 (conclusie) het al correct doet**: "Three novel ... techniques ... **are estimated to reduce** $K{=}9$ runtime by approximately 73%". Lijn r92 op r549. |

### MINOR

| # | Regel(s) | Probleem | Fix |
|---|----------|----------|-----|
| G2 | 795 | **`ET-Miner` (hoofdletter M)** in algorithm-caption — enige uitschieter tussen 40+ `ET-miner`. Zichtbaar in een float-caption. | `ET-miner` |
| G3 | 121, 169, 191, 204 | Bare `K-level(s)` zonder math-mode, terwijl de rest `$K$-level` gebruikt (r191 heeft beide vormen in dezelfde alinea). | `$K$-level(s)` |
| G4 | 92, 121, 130, 549 | Inconsistente eenheid-spatiëring bij uitgeschreven tijd: r121 `7.3~minutes` (beschermd), r92/r130/r549 plain spatie ("7.3 minutes", "63 minutes", "4.3 hours"). Huis-stijl is `\,` (80\,GB). | Standaardiseer naar `7.3\,minutes`, `63\,minutes`, `4.3\,hours`. |
| G5 | 102 | Lange appositie "structure *mining*, the systematic discovery ..., remains" leest als comma-splice. | Em-dashes: "structure *mining*---the systematic discovery of ...---remains". |
| G6 | 559 | **`Opus4.6`** mist spatie (author-info). | `Opus 4.6` |
| G7 | 92 vs 951 | Techniek heet "warp-cooperative" (abstract) vs "Warp-level" (appendix-tabel). | Eén label kiezen. |
| G8 | 873 vs 842 | Approx-symbool + eenheid-spatiëring wisselt: `$\sim$12 bytes` (873) vs `${\sim}12$~bytes` (842). | Standaardiseer naar `${\sim}12$\,bytes`. |

---

## Cross-check tegen eerdere reviews (feb–mei 2026)

De vier eerdere review-docs (`senior_review_mar23.md`, `review_b1_hostile.md`,
`review_b2_results.md`, `revision_notes_b3.tex`) draaiden op een **oudere** versie. Status nu:

| Eerder gemeld issue | Status in huidige `.tex` |
|---------------------|--------------------------|
| Table 1 feature-breakdown "fabricated" (247/752/3) | ✅ **GEFIXT** — toont nu 500/500/2 (r143–150). |
| "A fortiori"-argument logisch omgekeerd | ✅ **OPGELOST** — "a fortiori" is verwijderd; r437 erkent nu expliciet dat het null-model alleen op de Power-drempel draaide. |
| pLDDT-bins 3 vs 2 | ✅ **GEFIXT** — Run 1 = 2 bins (r150). |
| "GPU-resident" terminologie-fout | ✅ **ADRESSEERD** — "GPU-resident" (data) en "GPU-accelerated" (pad) worden nu correct als aparte termen gebruikt. |
| 5 permutaties statistisch zwak | ⚠️ **DEELS OPEN** — zie N4 (de p<0.17-claim is nog niet hard onderbouwd). |
| ⚠️ **NIEUW / TE BEVESTIGEN** | `revision_notes_b3.tex` noteerde voor de 35K-run @ 0.001%: **606,292 itemsets, K_max=17, 654 s**. De huidige paper claimt voor dezelfde run **16,8 miljard itemsets t/m K=8, 4.3 uur**. Dat is ~27.000× verschil — vrijwel zeker het effect van de eerder genoemde **NCCL int32-overflow bugfix** (pre-fix undercount). **Bevestig dat de huidige 16.8B de post-bugfix canonieke cijfers zijn** (intern is de huidige paper consistent — cumulatieve som klopt naar 16.8B). |

---

## Aanbevolen fix-prioriteit

1. **R1–R3** (foute auteurs/pagina's: barrio2024→Lau, coin2009→Terrapon, varadi2022→D439–D444) — citatie-integriteit, een editor pikt dit eruit.
2. **G1** (kapotte abstract-zin) — eerste indruk; fix bestaat al in r549.
3. **R4** (BIGMiner-schaal feitelijk fout én contraproductief voor je framing).
4. **N1–N2** (geheugen-conventies labelen — CSR 3/5.1/19 GB, dense 206 vs 26 GB).
5. **N4** (p<0.17 onderbouwen of naar ~0.45 corrigeren).
6. **R5–R13, N3, N5–N9, G2–G8** — losse details + housekeeping (orphans, `abramson2024` inzetten bij r537).

---

*Geen bestanden gewijzigd behalve dit rapport. Bron-URL's per referentie-fout opgenomen voor
directe verificatie. Subagent-IDs voor doorvragen: refs `aa32777a054d78988`, getallen
`a9269837d3627aafd`, taal `a9cfa6b2f9d788071`.*
