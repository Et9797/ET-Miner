# Data Mining the AlphaFold Protein Universe: How a 1994 Algorithm Found 26.8 Million Co-occurrence Patterns in 77 Million Proteins

*Rewrite Apriori as bit-vector arithmetic, put the whole transaction table on a GPU, and one of the oldest algorithms in data mining reads the largest protein database ever assembled in minutes.*

---

In 1994, Rakesh Agrawal and Ramakrishnan Srikant published Apriori, the algorithm that taught retailers that people who buy diapers also buy beer. Thirty years later, the same idea, applied to a very different kind of shopping basket, turns out to be a surprisingly good way to ask a big question: across every protein that AlphaFold has predicted a structure for, which annotations keep showing up together, and how deep do those combinations go?

This post explains the trick that makes that computation feasible, a vectorized reformulation of Apriori, and walks through what it found in the AlphaFold Protein Structure Database. The numbers below come from an independent re-execution of the published pipeline on 2 September 2026 (details in the reproducibility note at the end).

## Proteins as shopping baskets

Apriori works on *transactions*: sets of *items*. A transaction is frequent-itemset mining's unit of observation, and an itemset is frequent if it appears in at least a minimum number of transactions, the *support threshold*.

The mapping to proteins is direct:

- **One transaction per protein.** The AlphaFold DB (v4) has 214,683,829 entries. 205,620,298 of them have a mean predicted confidence (pLDDT) of at least 50.
- **Items are annotations.** For each protein, its Pfam domain families and Gene Ontology (GO) terms are taken from UniProt TrEMBL, plus one item for its confidence bin (mean pLDDT 50–90 or above 90).
- **A small, dense vocabulary.** Only the 500 most frequent Pfam families and the 500 most frequent GO terms (out of 24,291 and 25,993 seen) are kept, giving 1,006 defined items of which 1,002 actually occur.

After that encoding, 76,890,945 proteins (37.4%) carry at least two items. Those are the transactions that get mined; the other 128.7 million carry only their confidence bin and cannot participate in any pattern. The average mined protein has about four items, the busiest has 46.

The question then becomes: for a threshold as low as *eight proteins*, enumerate every combination of items that co-occurs at least that often. Combinations of size *K* are called *K*-itemsets, and the interesting ones are deep.

## Classic Apriori: scan, count, prune, repeat

The original algorithm is level-wise:

1. Count every single item; keep the frequent ones (*K* = 1).
2. Build candidate pairs from frequent items; scan the whole database counting how many transactions contain each candidate; keep the frequent ones (*K* = 2).
3. Build candidate triples by joining frequent pairs that share a prefix, prune any candidate with an infrequent subset (the *Apriori property*: a superset of an infrequent set cannot be frequent), scan the database again, keep the survivors.
4. Repeat until a level produces nothing.

Everything expensive lives in step "scan the database". The data are stored *horizontally*, one transaction after another, and each level walks through all of them, matching every transaction's items against the candidate set, classically with a hash tree. With 77 million transactions, an average of four items each, and millions of candidates per level, that is a lot of pointer chasing, and it happens once per level for twenty-two levels.

Later algorithms changed the layout. Eclat stores the database *vertically*: for each item, the list of transaction IDs that contain it. The support of an itemset is then the size of the intersection of its items' lists, and intersections compose: the list for {A, B, C} is the intersection of the list for {A, B} with the list for C. When data are dense enough, the ID lists become bitmaps, and intersection becomes a bitwise AND.

## The vectorized formulation

ET-Miner pushes the vertical, bitmap idea to its logical end and fits it to a GPU.

**The database is a bit matrix.** Every item gets a bit-vector with one bit per transaction: bit *t* is set if protein *t* carries the item. With 1,002 items and 76,890,945 transactions, each vector is 1,201,422 64-bit words and the entire database is 1,002 × 1,201,422 × 8 bytes ≈ 9.6 GB. It fits in the memory of a single modern GPU, so it is uploaded once and never touched from the host again. (A dense boolean table of the same data would be 77 GB; for the full 205.6-million set, 206 GB versus 25.8 GB as bit-vectors.)

**Support is a population count.** The support of an itemset is the number of set bits in the AND of its items' vectors. On a GPU, that is a stream of 64-bit AND and `popcount` instructions over 1.2 million words, followed by a reduction. There is no per-transaction matching, no hash tree, no branching on data; it is memory-bandwidth-bound arithmetic, which is exactly what a GPU is built for.

**Level 1** is one popcount per item.

**Level 2** is a fused kernel over all 501,501 item pairs: for pair (*i*, *j*), AND the two vectors, popcount, compare with the threshold, and write the pair out only if it survives. Counting and filtering happen in the same pass; nothing infrequent ever reaches memory.

**Levels 3 and up** use the same join as classic Apriori: candidate *K*-itemsets are formed from frequent (*K*−1)-itemsets that share a common (*K*−2)-prefix. The vectorized version exploits that structure: the AND of the shared prefix is computed once per *prefix group* and reused for every extension, so a candidate costs one additional AND-and-popcount rather than *K* of them. Candidates are laid out so that a GPU thread block handles one prefix group, results are compacted in place, and the level's survivors are sorted on the GPU to seed the next level. The Apriori property is still what keeps candidate sets small; it is just applied to sorted integer arrays instead of hash trees.

**Two small but important details.** First, a candidate whose running AND becomes all zeros can be abandoned early. Second, the loop condition is the same as the original: continue to level *K* while at least *K* frequent itemsets exist at level *K*−1. The algorithm stops on its own when the data run out of depth, with no cap on *K*.

The difference from the traditional formulation, in one sentence: classic Apriori spends its time moving through transactions to find candidates in them, while the vectorized formulation spends its time on candidates and answers each one with a fixed-size bit operation over all transactions at once. The database layout (vertical bitmaps instead of horizontal transaction lists) is what converts a data-structure-heavy search into dense arithmetic, and the fused count-and-filter kernels are what make the arithmetic cheap enough to run twenty-two levels deep on a 77-million-row table.

The result is exact. There is no sampling and no approximation: every itemset with support at or above the threshold is enumerated, which is also what makes the outcome checkable, as we will see.

## What it found

The published campaign runs the miner at six thresholds. The re-execution ran all six exhaustively; the counts below are the fresh ones.

| Run | Minimum proteins | Frequent itemsets | Deepest *K* |
|---|---|---|---|
| Base | 76,891 (0.1%) | 5,305 | 9 |
| Super | 7,690 (0.01%) | 113,405 | 14 |
| Power | 769 (0.001%) | 475,865 | 14 |
| Blitz | 77 | 2,841,280 | 19 |
| Ultra | 16 | 14,558,875 | 20 |
| Opus | 8 | 26,849,505 | 22 |

Four of the six rows match the published numbers exactly. The other two, Super and Power, are the rows for which the paper reported counts from a streaming variant of the miner (51,124 and 22,846 itemsets); the exhaustive counts are 113,405 and 475,865, and the paper's own controlled comparison already gave 475,865 for the exhaustive run at the Power threshold. More on that in the reproducibility note.

The deepest run is the headline: with a threshold of just eight proteins, the miner enumerates **26,849,505** frequent itemsets. Their size distribution is a broad hump that peaks at *K* = 9 (3,529,257 itemsets, 13.1% of the total) and thins out to 255 itemsets at *K* = 20, 23 at *K* = 21, and a single itemset at *K* = 22.

That single 22-item pattern is shared by exactly eight proteins, and each of the eight carries exactly those 22 items and nothing else, so this particular pattern cannot grow. Its members read like the résumé of a neuronal antiviral RNA helicase: the two DEAD/DEAH-box helicase domains (Pfam PF00270 and PF00271), ATP binding, RNA and DNA helicase activity, single- and double-stranded nucleic-acid binding, innate immune response, defense response to virus, cellular response to heat, and a spread of locations from cytosol and nucleus to mitochondrion, axon, dendrite, and nuclear speck, together with the medium-confidence structure bin. None of its 19 GO terms is an ancestor of another, so the depth is not an artifact of GO's hierarchy.

Is 22 a hard ceiling? Not by construction. Thirty-two proteins in the mined set carry 22 or more vocabulary items, nineteen carry 23 or more, and one carries 46. The ceiling is empirical: lowering the threshold from eight proteins to four (48,007,493 itemsets, 76 minutes on two 3090s) still ends at *K* = 22, with 342 patterns at *K* = 20, 27 at *K* = 21, and the same single pattern at *K* = 22. A 23-item pattern would need at least four of those nineteen proteins to agree on 23 items, and none do.

A few of the intermediate patterns are easier to recognize:

- A 19-item RNA-helicase/spliceosome signature shared by 187 proteins.
- A 17-item receptor-tyrosine-kinase signature (kinase domain PF07714 with SH2 PF00017 and SH3 PF00018) shared by 611 proteins.
- A 12-item bacterial cell-wall synthase signature (transpeptidase PF00905 with transglycosylase PF00912) in roughly ten thousand proteins (several 12-item variants exist, the largest in 16,185 proteins).

## Is this just what any big table would produce?

A permutation test says no. Shuffling the annotations among proteins while preserving how many annotations each protein has, then re-mining at the 769-protein threshold, gives about 171,000 frequent itemsets per shuffle against 475,865 in the real data. More telling than the totals is the depth. In 100 shuffles (the paper ran five; the re-execution ran a hundred with the released code), not one produced an itemset beyond *K* = 6, whereas the real data contain 88,745 itemsets of size seven or more. With 100 shuffles the empirical p-value for that observation is about 0.01, and the 95% upper bound on the chance that a random shuffle reaches *K* = 7 is 3%. The deep combinations are properties of biology and of how curators annotate it, not of the table's marginals.

## What it costs

On the GPU the paper used (one NVIDIA H100, February 2026), the deepest run took 7.3 minutes; that figure was not re-measured. The re-execution used two consumer RTX 3090 cards. The same run takes 19 minutes on one of them with the single-GPU kernels (16 of those in the mining loop itself), or 44 minutes with the two-GPU row-split path that streams every level to disk. The whole six-threshold campaign takes 38 minutes on one 3090, and the hundred-permutation null model 33 minutes on both. Feature extraction, the unglamorous part, reads a 150 GiB compressed UniProt flat file and the AlphaFold metadata table. The paper reports 63 minutes for it; the re-execution streamed the archive member and trimmed it to the lines the extractor needs (48 minutes, mostly network time), ran two more filtering passes (13 and 16 minutes) because this machine has less memory than the paper's, and then extracted in 10 minutes, so the two figures are not comparable.

## What it does not mean

These are co-occurrences of annotations, mined at a chosen threshold, over a vocabulary of the 1,000 most common labels. They are not mechanisms, and they inherit every bias of how UniProt's automatic annotation works. The value of the exercise is different: it turns "which combinations of labels exist, how often, and how deep" into a question with an exact, reproducible answer over the entire annotated AlphaFold universe, computed in the time it takes to make coffee.

## Reproducibility note

Every count above was re-derived on 2 September 2026, on a rented machine with two RTX 3090s, from a fresh export of the AlphaFold DB metadata, a fresh download of UniProt TrEMBL, a fresh feature extraction, and fresh mining runs with the released ET-Miner code. Every deterministic number in the paper that the re-execution could test came back identical: the protein counts (214.7 million entries, 205.6 million with pLDDT ≥ 50, 76.9 million with two or more items), the vocabulary, the four exhaustive itemset counts, the full 22-level size distribution, the 22-item pattern and its eight proteins. Four things did not survive.

- **The UniProt release.** The paper states release 2025_01. The record count in the original logs identifies release 2026_01, and only that release reproduces the results.
- **The streaming-mode numbers.** The paper's Super and Power rows came from a streaming (SON) mode of the miner and reported 51,124 and 22,846 itemsets, plus a claim that streaming misses 95% of patterns. The released streaming code is exact: at the Power threshold it returned the same 475,865 itemsets as the exhaustive run, 62 times slower. The approximate-mode numbers are historical; the exhaustive counts replace them.
- **The argument for the ceiling.** The paper argued that *K* = 23 is impossible because no protein carries more than 22 vocabulary items. That premise is false (one protein carries 46). The ceiling stands, but as an empirical result, as described above.
- **The five-shuffle statistics.** The paper's per-size Z-scores from five shuffles (+3,791, +7,402, +71,728) could not be regenerated: the draws were not recorded, and five fresh draws with the released code give +3,048, +9,917 and +175,697. What does reproduce is the structure, no shuffle reaching *K* = 7; the hundred-shuffle numbers above replace them.

Timings were measured on different hardware from the paper's and are reported as such.

---

*Apriori: Agrawal & Srikant, "Fast Algorithms for Mining Association Rules", VLDB 1994. AlphaFold DB: Varadi et al. 2024. ET-Miner code: github.com/Et9797/ET-miner.*
