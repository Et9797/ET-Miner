"""Consolidate the V2 citation audit into citation_audit.json and CITATION_AUDIT.md.

The per-claim verdicts (verbatim quotes, grep anchors, retrieval notes) come from the five
verification agents (agent_verdicts/*.json); the bibliographic checks from crossref_check.py
(crossref_check.md). This file holds the consolidated decision per bibliography key and writes
both outputs next to itself.

Usage:
    python make_citation_audit.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

METHOD = (
    "Two checks per bibliography entry. (1) Metadata: every entry was matched against CrossRef and "
    "Semantic Scholar by title and first author (`crossref_check.py` -> `crossref_check.md`), with DBLP "
    "and proceedings PDFs for the conference papers CrossRef does not index. (2) Claim support: every "
    "sentence carrying a `\\cite` was checked against the cited work's retrieved text by five "
    "verification agents working from full texts where open (Europe PMC, PLOS, BMC, vldb.org, "
    "ceur-ws.org, ijcai.org, arXiv, author copies) and from abstracts otherwise; their verdicts with "
    "verbatim quotes and grep anchors are in `agent_verdicts/*.json`. The retrieved texts stayed in the "
    "session scratchpad because most are copyrighted; the quotes below are verbatim. Actions: VERIFIED "
    "claims were kept; NOT SUPPORTED and UNVERIFIABLE claims were deleted or cut back to the words the "
    "source states; values stated by a different work already in the bibliography were re-attributed; "
    "the tool-comparison values (Introduction, Discussion 4.2 with Table 6, Appendix D Tables 7 and 8, "
    "Appendix C.1) were deleted regardless of verifiability, as the V2 brief required; entries left "
    "uncited were dropped from the bibliography, which was then reordered to first-citation order."
)

E = []


def add(key, doi, bib, outcome, claims):
    E.append({"key": key, "doi": doi, "bib_check": bib, "outcome": outcome, "claims": claims})


def c(where, claim, verdict, evidence, action):
    return {"where": where, "claim": claim, "verdict": verdict, "evidence": evidence, "action": action}


add("jumper2021", "10.1038/s41586-021-03819-2", "ok", "kept",
    [c("Introduction", "AlphaFold is the method behind the database", "VERIFIED",
       "\"Here we provide the first computational method that can regularly predict protein structures with atomic accuracy\" (jumper2021.txt, Europe PMC full text)", "kept")])
add("varadi2022", "10.1093/nar/gkab1061", "ok", "kept",
    [c("Introduction", "over 200 million proteins", "NOT SUPPORTED",
       "\"The initial release of AlphaFold DB contains over 360,000 predicted structures across 21 model-organism proteomes\" (varadi2022.txt)",
       "figure re-attributed: varadi2024 added to the citation group; entry kept for the database itself"),
     c("Introduction", "spanning virtually all known organisms", "NOT SUPPORTED",
       "no retrieved source states it (varadi2024 lists exclusions; TED gives 'more than 1 million taxa')", "phrase deleted")])
add("varadi2024", "10.1093/nar/gkad1011", "title corrected: subtitle 'providing structure coverage for over 214 million protein sequences' restored", "kept",
    [c("Methods 2.1, Introduction, Discussion 4.2", "214 million predicted structures across UniProt", "VERIFIED",
       "\"The database provides access to over 214 million predicted structures\" (varadi2024.txt)", "kept")])
add("uniprot2023", "10.1093/nar/gkac1052", "ok", "kept",
    [c("Methods 2.1", "annotations from the 'UniProt cross-reference pipeline'", "NOT SUPPORTED as worded",
       "no occurrence of 'Pfam' or 'cross-reference' in the paper; the extractor reads the `DR   Pfam;` and `DR   GO;` lines of the flat file (runs/20260902T0000Z/phase2/scripts/filter_dat_pfamgo.sh)",
       "wording corrected to 'the Pfam and GO cross-reference lines of the UniProt flat file'"),
     c("Limitations item 3", "UniProt is released every eight weeks", "VERIFIED",
       "\"UniProt releases are published every eight weeks.\" (uniprot2023.txt)", "added with citation")])
add("mistry2021", "10.1093/nar/gkaa913", "ok", "kept",
    [c("Methods 2.1", "Pfam domain assignments", "VERIFIED", "\"The Pfam database is a widely used resource for classifying protein sequences into families and domains.\" (mistry2021.txt)", "kept")])
add("go2023", "10.1093/genetics/iyad031", "ok", "kept",
    [c("Methods 2.1", "Gene Ontology terms", "VERIFIED", "GO knowledgebase paper (go2023.txt)", "kept"),
     c("Limitations item 3", "GO is released monthly", "VERIFIED",
       "\"citable, versioned updates are released on a monthly basis.\" (go2023.txt); geneontology.org: \"GO has monthly releases\"",
       "previously uncited 'GO annotations are updated monthly' now reads 'The Gene Ontology is released monthly' with this citation")])
add("wang2011", "10.1371/journal.pone.0017906", "ok", "kept",
    [c("Introduction, Discussion 4.2", "domain co-occurrence networks, pairwise, scale-free", "VERIFIED",
       "\"Two domain types (i.e. nodes) are connected by an edge if they co-exist in one protein\"; \"These DCNs have the hallmark features of scale-free networks.\" (wang2011.txt)", "kept"),
     c("Discussion 4.2", "limited to individual proteomes", "NOT SUPPORTED",
       "one DCN per organism, built for 398 prokaryotes and 19 eukaryotes and aligned across species (wang2011.txt)", "clause deleted")])
add("coin2009", "10.1093/bioinformatics/btp560", "author list completed (N. Terrapon, O. Gascuel, E. Marechal, L. Brehelin)", "kept",
    [c("Introduction, Discussion 4.2", "statistical co-occurrence of domain pairs for new-domain detection in P. falciparum", "VERIFIED",
       "\"extract domain pairs showing strong co-occurrence (as assessed by a statistical test)\"; \"Applied to P. falciparum, our method identifies 585 new Pfam domains\" (coin2009_flow.txt, HAL author manuscript)", "kept")])
add("meysman2015", "10.1186/s13040-015-0038-4", "ok", "kept",
    [c("Introduction, Discussion 4.2", "spatially cohesive amino-acid patterns from the PDB; ~32K structures", "VERIFIED",
       "\"contains 32 142 protein molecules from a large variety of organisms\"; cohesive radius 4.5 A between C-alpha atoms (meysman2015.txt)", "kept")])
add("mrzic2018", "10.1186/s13040-018-0181-9", "ok", "kept",
    [c("Discussion 4.2", "reviewed frequent subgraph mining for biomolecular data", "VERIFIED", "title and scope (mrzic2018.txt)", "kept"),
     c("Discussion 4.2", "graph mining is NP-hard and scales poorly", "NOT SUPPORTED",
       "'NP' does not occur; the review says \"Frequency counting is usually the most computationally intensive part of subgraph mining algorithms.\" (mrzic2018.txt)", "clause deleted")])
add("barrio2024", "10.1126/science.adq4946", "ok (authors Lau et al.; the key name is cosmetic)", "kept",
    [c("Discussion 4.2", "TED catalogues domains across the AlphaFold Database", "VERIFIED",
       "\"comprehensive analysis of domain composition for the entirety of the AFDB (version 4)... over 364 million putative domains, derived from more than 214 million protein sequences\" (barrio2024.txt)", "kept")])
add("rehwinkel2020", "10.1038/s41577-020-0288-3", "ok", "kept",
    [c("Results 3.3", "RIG-I and MDA5 are established viral RNA sensors in innate immunity", "VERIFIED",
       "\"RLRs are key sensors of virus infection\" (rehwinkel2020.txt, PMC author manuscript)", "kept"),
     c("Results 3.3", "described as DEAD-box helicases", "NOT SUPPORTED",
       "\"form a subfamily of SF2 helicases\"; \"central DECH-box helicase domain\"; 'DEAD' never occurs (rehwinkel2020.txt)", "wording changed to 'The SF2 helicases RIG-I and MDA5'")])
add("wahl2009", "10.1016/j.cell.2009.02.009", "ok", "kept",
    [c("Results 3.4 (K=19)", "DEAD/DEAH-box helicases are core pre-mRNA splicing machinery", "VERIFIED",
       "\"Eight evolutionarily conserved DExD/H-type RNA-dependent ATPases/helicases act at specific steps of the splicing cycle\" (wahl2009.txt, publisher PDF via Wayback)", "kept")])
add("boggon2004", "10.1038/sj.onc.1208081", "ok", "kept",
    [c("Results 3.4 (K=17)", "Src-family kinases: SH3, SH2 and kinase domains", "VERIFIED (abstract)",
       "\"Their conserved domain organization includes a myristoylated N-terminal segment followed by SH3, SH2, and tyrosine kinase domains\" (boggon2004_abstract.txt)", "kept")])
add("dillingham2008", "10.1128/MMBR.00020-08", "ok", "removed (uncited after the edit)",
    [c("Results 3.4 (K=13)", "RecBCD repairs double-strand breaks by recombination; SOS", "VERIFIED",
       "\"recombinational DNA repair\"; \"a loss of SOS induction\" in recB/recC mutants (dillingham2008_flow.txt)", "-"),
     c("Results 3.4 (K=13)", "a DEAD-box helicase pattern is 'consistent with RecBCD-like repair complexes'", "NOT SUPPORTED",
       "RecB carries \"motifs characteristic of superfamily 1 (SF1) DNA helicases\"; the review never links DEAD-box (SF2) helicases to RecBCD (dillingham2008_flow.txt)", "clause deleted; entry removed")])
add("sauvage2008", "10.1111/j.1574-6976.2008.00105.x", "ok", "kept",
    [c("Results 3.4 (K=12)", "PBPs carry transpeptidase and transglycosylase domains and act in peptidoglycan biosynthesis", "VERIFIED",
       "\"C-terminal penicillin-binding domain of both classes has a transpeptidase (TP) activity\"; \"N-terminal domain is responsible for their glycosyltransferase activity\" (sauvage2008.txt, CSIC author manuscript)", "kept"),
     c("Results 3.4 (K=12), Discussion 4.3", "beta-lactam antibiotics bind 'these exact domains' / 'these exact domain combinations' (both domains)", "NOT SUPPORTED",
       "\"penicillin-binding (PB) domain, which binds beta-lactam antibiotics (figure 2)\"; the glycosyltransferase domain is the moenomycin target (sauvage2008.txt)",
       "cut to 'bind the penicillin-binding domain that carries the transpeptidase activity' (3.4) and 'bind the transpeptidase (penicillin-binding) domain of these proteins' (4.3)")])
add("agrawal1994", "none (VLDB 1994, pp. 487-499; DBLP + proceedings PDF)", "ok", "kept",
    [c("Introduction", "Apriori discovers all itemsets above a minimum support level-wise", "VERIFIED",
       "\"Itemsets with minimum support are called large itemsets\"; \"Algorithms for discovering large itemsets make multiple passes over the data.\" (agrawal1994.txt, vldb.org)", "kept")])
add("han2000", "10.1145/342009.335372", "ok", "kept",
    [c("Methods 2.2", "FP-Growth / FP-tree", "VERIFIED",
       "\"we propose a novel frequent pattern tree (FP-tree) structure... FP-growth, for mining the complete set of frequent patterns\" (han2000.txt); the 'inherently sequential' clause is the authors' own argument and is not attributed", "kept")])
add("savasere1995", "none (VLDB 1995, pp. 432-444; DBLP + proceedings PDF)", "ok", "kept",
    [c("Methods 2.5, Results 3.1, Discussion 4.1", "partition the database, mine each partition locally, global counting pass; a globally frequent itemset is locally frequent in some partition", "VERIFIED",
       "\"Partition algorithm accomplishes this in two scans of the database.\"; \"any potential large itemset appears as a large itemset in at least one of the partitions\"; \"a cumulative count over all partitions gives the support for an itemset in the entire database.\" (savasere1995.txt, vldb.org)", "kept"),
     c("Methods 2.5, Discussion 4.1", "mined at a 'lowered' local threshold; algorithm called 'SON'", "NOT SUPPORTED",
       "local large itemsets use the same user-defined minimum support per partition; the word SON does not occur (the paper calls it the Partition algorithm)",
       "'lowered' deleted; named 'the Partition algorithm of Savasere, Omiecinski and Navathe'; the lowered threshold stated as ET-miner's own choice")])
add("miettinen2020", "10.24963/ijcai.2020/685", "ok", "kept",
    [c("Introduction", "Boolean matrices are central to pattern-discovery and Boolean matrix methods", "VERIFIED",
       "BMF survey; \"(they also correspond to closed itemsets in frequent itemset mining)\"; \"(i.e. they generate a tiling)\" (miettinen2020.txt, ijcai.org)", "kept")])
add("phipson2010", "10.2202/1544-6115.1585", "ok", "kept",
    [c("Table 5 caption, Results 3.5", "+1 correction; resolution 1/(m+1)", "VERIFIED",
       "\"a biased estimator (b + 1)/(m + 1)\"; \"P(p-hat <= alpha) is never less than 1/(m + 1)\" (phipson2010.txt, arXiv 1603.05766)", "kept")])
add("webb2007", "10.1007/s10994-007-5006-x", "ok", "kept",
    [c("Limitations item 1", "framework for statistically significant pattern discovery", "VERIFIED (abstract)",
       "\"enforce a strict upper limit on the risk of experimentwise error\" (webb2007.txt)", "kept")])
add("webb2014", "10.1145/2601433", "ok", "kept",
    [c("Limitations item 1", "framework for statistically significant pattern discovery", "VERIFIED (abstract)",
       "\"pruning mechanisms based on upper bounds on itemset value and statistical significance level\" (webb2014.txt)", "kept")])
add("abramson2024", "10.1038/s41586-024-07487-w", "ok", "removed (uncited after the edit)",
    [c("Future work", "the AlphaFold database expanded to include predicted complexes", "NOT SUPPORTED",
       "the AlphaFold 3 paper never mentions the AlphaFold Database or added homodimers (abramson2024.txt, full text)", "citation replaced by afdb2026news and han2026afdb; entry removed")])
add("afdb2026news", "https://www.ebi.ac.uk/about/news/technology-and-innovation/first-complexes-alphafold-database/ (16 March 2026, updated 19 May 2026)", "new entry", "added",
    [c("Future work", "1.7 million high-confidence homodimers added; 18 million lower-confidence for bulk download", "VERIFIED",
       "\"1.7 million high-confidence homodimer predictions have been added to the AlphaFold Database. Another 18 million are lower-confidence homodimers, which are available for bulk download from the EMBL-EBI FTP server.\" (embl_ebi_homodimer_news.txt)", "kept, now cited"),
     c("Future work", "19 May 2026 update: almost 80,000 high-confidence heterodimers, 8.1 million lower-confidence", "VERIFIED",
       "\"Update (19 May 2026): Almost 80,000 high-confidence heterodimer predictions have been added to the AlphaFold Database. A further 8.1 million lower-confidence heterodimer predictions are available for bulk download.\"",
       "replaces the outdated 'Heterodimer predictions are being assessed and will follow'"),
     c("Future work", "quotation 'a first step towards a comprehensive description of the human interactome'", "VERIFIED",
       "verbatim, inside the quotation attributed to Dame Janet Thornton (Director Emeritus, EMBL-EBI)", "attributed to Janet Thornton with the citation")])
add("han2026afdb", "10.64898/2026.03.27.714458 (bioRxiv, posted 29 March 2026)", "new entry (CrossRef: Han, Tsenkov, Venanzi, ...)", "added",
    [c("Future work", "the AlphaFold Database expanded to proteome-scale quaternary structures", "VERIFIED",
       "title and abstract (biorxiv_afdb_complexes.txt)", "cited for the expansion; the numbers are cited to the announcement only")])
add("chothia2003", "10.1126/science.1085371", "ok", "kept",
    [c("Discussion 4.3", "modular architecture of domains; evolution reuses and combines a finite set of building blocks", "UNVERIFIABLE",
       "full text closed (science.org 403, no open copy); abstract: \"Most proteins have been formed by gene duplication, recombination, and divergence.\" (chothia2003.txt)",
       "sentence cut to 'the formation of most proteins by gene duplication, recombination and divergence'")])
add("rhee2008", "10.1038/nrg2363", "ok", "kept",
    [c("Discussion 4.3", "electronically propagated GO annotations may create artificial co-occurrence", "UNVERIFIABLE",
       "full text closed; Key Points: evidence codes \"are often overlooked\"; pitfalls include \"indiscriminate propagation of annotations through the hierarchy, and ignoring the correlations between GO terms\" (rhee2008.txt)",
       "sentence cut to the Key Points: evidence codes overlooked; propagation through the hierarchy and correlations between terms are known pitfalls")])
add("luna2019", "10.1002/widm.1329", "article number e1329 was missing", "removed (Introduction sentence deleted)",
    [c("Introduction", "GPU FIM systems achieved 50-350x speedups", "NOT SUPPORTED",
       "a survey with no numeric GPU speedup; \"some novel parallel GPU-based approaches were proposed\" (luna2019.txt, author copy)", "sentence deleted (tool comparison)")])
add("zaki2000", "10.1109/69.846291", "ok", "removed (Introduction sentence deleted)",
    [c("Introduction", "Eclat", "VERIFIED", "\"These include Eclat (Equivalence CLAss Transformation)...\" (zaki2000.txt)", "sentence deleted (tool comparison)")])
add("zaki1997", "none (KDD-97)", "pages were wrong: 283-286, not 283-296 (DBLP and the AAAI PDF)", "removed (Introduction sentence deleted)",
    [c("Introduction", "Eclat", "VERIFIED", "\"Eclat: equivalence class & bottom-up\" (zaki1997_aaai.txt)", "sentence deleted (tool comparison)")])
add("zhang2011", "10.1109/CLUSTER.2011.61", "ok", "kept for one verified statement (Methods 2.5, pass 10: candidates generated on the CPU, copied to the GPU, support values copied back; item bitsets GPU-resident)",
    [c("Discussion 4.2 (old)", "up to 100x on FIMI benchmarks with static bitsets", "VERIFIED",
       "\"demonstrate up to 100X speedup as compared with several state-of-the-art FIM algorithms on a CPU\"; \"static bitset\" (zhang2011.txt, author copy)", "deleted (tool comparison)"),
     c("Methods 2.5", "transfers candidate sets in and support arrays out at each level", "VERIFIED",
       "\"candidates are... copied from main memory to graphic memory by host code... and the support value results are copied back to main memory\"", "clause deleted with the unquantified 'orders of magnitude less'"),
     c("Appendix C.3", "re-transfers transaction bitmaps at each iteration", "NOT SUPPORTED",
       "\"Only the vertical lists of first generation will be saved in graphics memory\" (bitsets stay GPU-resident)", "clause deleted"),
     c("Introduction", "50-350x over Eclat and FP-Growth", "NOT SUPPORTED",
       "\"4X-10X speed up... up to 80X\" versus Borgelt's Apriori only", "sentence deleted")])
add("chon2018", "10.1016/j.ins.2018.01.046", "ok", "removed (all citing clauses deleted)",
    [c("Introduction, Methods 2.5, Discussion 4.2, Tables 6-8, Appendix C", "streaming of bitmap chunks; 15M/20K/K~30/4x GTX 1080/20-150 s; 1.7M real; cost model; 32-bit __popc; per-iteration transfers", "UNVERIFIABLE",
       "full text not retrievable (ScienceDirect blocks non-browser access despite the CC BY-NC-ND licence; no repository copy; official host offline); abstract only: \"by orders of magnitude on the tested benchmarks\" (chon2018.txt)", "deleted (tool comparison)")])
add("chon2024", "10.1016/j.eswa.2024.123928", "authors are K.-W. Chon and C. Kim (two), title prefix 'GMiner++:' was dropped", "removed (all citing clauses deleted)",
    [c("Discussion 4.2 (old), Table 7", "pre-calculated bit arrays; CPU candidate generation; replicated bit array blocks", "VERIFIED",
       "\"decreased redundant computations using pre-calculated bit arrays with bit array blocks\"; \"a CPU generates candidates... whereas several GPUs count candidate occurrences\" (chon2024.txt, abstract and introduction)", "deleted (tool comparison)"),
     c("Appendix C.1, Table 7", "32-bit __popc", "UNVERIFIABLE", "not in the retrievable text", "deleted")])
add("chon2018b", "10.1007/s10586-018-1812-0", "ok", "removed (all citing clauses deleted)",
    [c("Discussion 4.2 (old), Table 6", "largest prior scale: 100M transactions on 30 MapReduce nodes; 100K items; 1-20K s", "NOT SUPPORTED",
       "abstract: \"large-scale datasets of up to 6.5 billion transactions\" (chon2018b.txt)", "deleted (tool comparison)")])
add("djenouri2019", "10.1016/j.ins.2018.07.020", "title was wrong: 'Exploiting GPU and cluster parallelism in single scan frequent itemset mining' (Djenouri, Djenouri, Belhadi, Cano)", "removed (all citing clauses deleted)",
    [c("Discussion 4.2 (old)", "combined GPU and cluster computing", "VERIFIED (abstract)",
       "\"(CGSS) accelerates the frequent itemset mining process by using multiple cluster nodes equipped with GPUs\" (djenouri2019.txt)", "deleted (tool comparison)"),
     c("Methods 2.5, Appendix C.3", "transfers candidate sets and bitmaps each iteration", "UNVERIFIABLE",
       "single-scan method; methods section paywalled", "clause deleted"),
     c("Introduction", "350x speedup over Eclat/FP-Growth", "NOT SUPPORTED",
       "\"up to a 350 times speedup\" is versus the authors' own sequential single-scan algorithm on a GPU cluster", "sentence deleted")])
add("acmsurvey2021", "10.1145/3472289", "title uses 'Frequent Itemsets Mining' (plural)", "removed (all citing clauses deleted)",
    [c("Introduction, Discussion 4.2 (old)", "Apriori-based approaches dominate GPU implementations; 50-350x", "UNVERIFIABLE",
       "abstract only (acmsurvey2021.txt)", "deleted (tool comparison)")])
add("fang2009", "10.1145/1565694.1565702", "author list was wrong: the paper has five authors (W. Fang, M. Lu, X. Xiao, B. He, Q. Luo), not the ten listed", "removed (all citing clauses deleted)",
    [c("Discussion 4.2 (old)", "foundational GPU-FIM work", "VERIFIED",
       "\"there has been no prior work that focuses on studying the GPU acceleration for FIM algorithms\" (fang2009.txt)", "deleted (tool comparison)"),
     c("Table 6", "100K transactions / 1K items / GTX 280 / max K ~5 / 1-10 s", "PARTLY SUPPORTED",
       "T40I10D100K with 1,000 items and 100,000 transactions on \"an NVIDIA GTX 280 GPU\" verified; no maximum K reported; times only in bar charts; and \"CPU-based FP-growth is faster than our both GPU-based implementations by a factor of 4 to 16\"", "deleted (tool comparison)")])
add("borgelt2003", "none (CEUR-WS Vol. 90)", "ok", "removed (all citing clauses deleted)",
    [c("Table 6", "100K transactions / 500 items / max K ~10 / 1-100 s", "NOT SUPPORTED",
       "datasets named only (BMS-Webview-1, T10I4D100K, census, chess, mushroom); no transaction/item counts or maximum size stated; times only as log-scale plot axes (borgelt2003.txt, ceur-ws.org)", "deleted (tool comparison)")])
add("naulaerts2015", "10.1093/bib/bbt074", "ok", "kept (sentence re-homed to Discussion 4.2)",
    [c("Discussion 4.2", "primer on frequent itemset mining for bioinformatics", "VERIFIED",
       "\"In this primer, we introduce frequent itemset mining and their derived association rules for life scientists.\" (naulaerts2015.txt, Europe PMC)", "the deleted 'systematic review' sentence of old 4.2 replaced by 'give a primer on frequent itemset mining for bioinformatics'")])

UNATTRIBUTED = [
    "Appendix C.1: '__popcll ... in a single clock cycle' had no source and is not stated by the CUDA documentation; deleted.",
    "Methods 2.5 / Discussion 4.1 / Appendix C.3: 'orders of magnitude less than conventional approaches' and 'versus full candidate and support array transfers in conventional approaches' had no artifact and no supporting source; deleted.",
    "Limitations item 3: 'GO annotations are updated monthly' was uncited; now cites go2023 (monthly GO releases) and uniprot2023 (eight-weekly UniProt releases).",
    "Results 3.4 (K=17): '~611 proteins' is the exact support of the single matching itemset (RESULTS P-030); the tilde was removed.",
    "Pass 10 (adversarial review): uncited attributions deleted — 'transfer overhead that dominates prior GPU FIM systems' (Introduction), 'FP-tree construction is inherently sequential' (Methods 2.2), the receptor-tyrosine-kinase / imatinib-dasatinib-erlotinib sentence (Results 3.4; boggon2004 describes non-receptor Src-family kinases and EGFR carries no SH2/SH3 domain), 'the bacterial equivalent of the eukaryotic ubiquitin-proteasome system' (Results 3.4), 'None of these approaches perform exhaustive itemset mining beyond K=2' (Discussion 4.2; Meysman mined triplets, Terrapon reports triplets and a quartet); 'Traditional approaches ...' (Methods 2.5) replaced by the verified GPApriori description with zhang2011 restored.",
    "Pass 10: the K=13 and K=11 level maxima (RESULTS P-030) are held by itemsets other than the described patterns (a penicillin-binding-protein itemset at K=11; an itemset without the SOS term at K=13); the text now says so and adds the pattern-specific maxima 10,978 (23 itemsets) and 18,257 (491 itemsets) from RESULTS P-031; headings changed to 'Helicase DNA Repair and Recombination Module' and 'AAA+ ATPase--Clp Protease Module', 'Receptor Tyrosine Kinase Signaling Hub' to 'Src-Family Kinase Signaling Module'.",
]

BIB_CORRECTIONS = [
    "varadi2024: full title restored",
    "coin2009: four authors listed instead of 'et al.'",
    "noted but moot because the entry was removed: zaki1997 pages 283-286; luna2019 article e1329; chon2024 two authors and 'GMiner++:' title prefix; djenouri2019 actual title and authors; fang2009 five authors; acmsurvey2021 'Itemsets'",
]
REMOVED = ["luna2019", "zaki2000", "zaki1997", "chon2018", "djenouri2019", "acmsurvey2021", "chon2024",
           "dillingham2008", "fang2009", "abramson2024", "chon2018b", "borgelt2003"]
ADDED = ["afdb2026news", "han2026afdb"]


def main():
    audit = {"generated": "2026-09-04", "method": METHOD, "entries": E, "unattributed_claims": UNATTRIBUTED,
             "bib_corrections": BIB_CORRECTIONS, "bibitems_removed": REMOVED, "bibitems_added": ADDED,
             "inputs": ["agent_verdicts/verdicts_A.json (AlphaFold, databases, statistics)", "agent_verdicts/verdicts_B.json (domain co-occurrence, pattern biology)",
                        "agent_verdicts/verdicts_C.json (classical FIM)", "agent_verdicts/verdicts_D1.json (GPApriori, Fang, CGSS, survey)",
                        "agent_verdicts/verdicts_D2.json (GMiner, GMiner++, BIGMiner)", "crossref_check.md (metadata)"],
             "tex_passes": ["../revise_tex_v2_pass8_citations.py", "../revise_tex_v2_pass9_citations.py", "../reorder_bibliography.py"]}
    (HERE / "citation_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {}
    for e in E:
        for cl in e["claims"]:
            v = cl["verdict"].split(" ")[0].split("(")[0]
            counts[v] = counts.get(v, 0) + 1
    out = ["# Citation audit of the V2 preprint (paper/et_miner_proteome.tex)\n",
           "Generated 2026-09-04 by `make_citation_audit.py` from the agent verdict files and the metadata check in this directory.\n",
           "## Method\n", METHOD + "\n",
           "## Summary\n",
           f"- {len(E)} bibliography keys examined (38 original, 2 added); {sum(len(e['claims']) for e in E)} claims checked.",
           f"- Claim verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".",
           f"- Entries kept: {sum(1 for e in E if e['outcome'].startswith('kept'))}; removed: {len(REMOVED)}; added: {len(ADDED)}. The bibliography now has 27 entries in first-citation order.",
           "- Every value describing another tool or system was deleted with the sentence or table that carried it (Introduction GPU-FIM sentence, Discussion 4.2 with Table 6, Appendix D Tables 7 and 8, the popcount comparison in Appendix C.1), as the V2 brief required, independently of the verdicts.\n",
           "## Per-entry outcomes\n",
           "| key | DOI / source | bibliographic check | outcome |", "|---|---|---|---|"]
    for e in E:
        out.append(f"| {e['key']} | {e['doi']} | {e['bib_check']} | {e['outcome']} |")
    out += ["", "## Claims, verdicts and evidence\n"]
    for e in E:
        out.append(f"### {e['key']}\n")
        out.append("| where | claim | verdict | evidence (verbatim quotes; file names refer to the retrieved texts) | action |")
        out.append("|---|---|---|---|---|")
        for cl in e["claims"]:
            row = [cl["where"], cl["claim"], cl["verdict"], cl["evidence"], cl["action"]]
            out.append("| " + " | ".join(x.replace("|", "/") for x in row) + " |")
        out.append("")
    out += ["## Unattributed claims handled in the same passes\n"] + [f"- {u}" for u in UNATTRIBUTED] + [""]
    out += ["## Bibliographic corrections\n"] + [f"- {b}" for b in BIB_CORRECTIONS] + [""]
    out += ["## Bibliography entries removed\n", ", ".join(REMOVED) + "\n", "## Bibliography entries added\n", ", ".join(ADDED) + "\n"]
    (HERE / "CITATION_AUDIT.md").write_text("\n".join(out), encoding="utf-8")
    print("entries:", len(E), "claims:", sum(len(e["claims"]) for e in E), "verdicts:", counts)


if __name__ == "__main__":
    main()
