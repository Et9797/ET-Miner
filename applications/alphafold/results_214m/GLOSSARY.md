# Biological Glossary - AlphaFold 214M Mining Results

Reference glossary for all Pfam protein domains and Gene Ontology (GO) terms discovered in the frequent itemset mining of 214 million AlphaFold-predicted protein structures.

---

## Pfam Domains

Protein family domains identified through InterPro/Pfam classification. Domains listed here appear in high-K itemsets (K>=10) from the 2.84M direct GPU mining results.

| ID | Name | Description |
|----|------|-------------|
| PF00001 | 7tm_1 (GPCR) | Seven-transmembrane G protein-coupled receptor, rhodopsin family. Detects extracellular molecules and activates intracellular G protein signaling cascades. ~34% of FDA-approved drugs target GPCRs. |
| PF00004 | AAA (ATPase) | ATPases Associated with diverse cellular Activities. Ring-shaped hexameric molecular motors that couple ATP hydrolysis to protein unfolding, disaggregation, and translocation. Includes cell-cycle regulators, proteolytic complex subunits, and molecular chaperones. |
| PF00017 | SH2 domain | Src Homology 2 domain (~100 aa). Binds phosphorylated tyrosine residues via a conserved Arg residue. Core signaling module that couples receptor tyrosine kinases to intracellular pathways. Over 100 human proteins contain SH2 domains. |
| PF00018 | SH3 domain | Src Homology 3 domain (~60 aa). Beta-barrel fold that binds proline-rich peptide motifs. Mediates assembly of signaling protein complexes in cytoskeleton regulation, Ras and Src kinase pathways. ~300 SH3 domains in the human proteome. |
| PF00115 | COX1 | Cytochrome c oxidase subunit I. Terminal enzyme of the mitochondrial electron transport chain. Catalyzes electron transfer from cytochrome c to molecular oxygen, coupled to proton pumping across the inner mitochondrial membrane. |
| PF00130 | C1_1 domain | Phorbol ester / diacylglycerol (DAG) binding domain (~50 aa). Contains two Zn2+-binding clusters. Found in protein kinase C and other signaling proteins; mediates membrane recruitment upon DAG production. |
| PF00270 | DEAD/DEAH helicase N-terminal | N-terminal RecA-like domain of DEAD/DEAH-box helicases containing the conserved Asp-Glu-Ala-Asp/His motif. ATP-dependent RNA/DNA unwinding core. Central roles in RNA metabolism, pre-mRNA splicing, ribosome biogenesis, and translation regulation. |
| PF00271 | Helicase C-terminal | Conserved C-terminal domain of SF1/SF2 helicases. Second RecA-like domain that completes the helicase motor. Works with PF00270 to unwind nucleic acid duplexes in an ATP-dependent manner. Essential for DNA repair, recombination, and replication. |
| PF00621 | RhoGEF domain | Rho guanine nucleotide exchange factor (Dbl homology) domain. Activates Rho family GTPases by catalyzing GDP-to-GTP exchange. Regulates cytoskeletal dynamics, cell morphology, migration, and intracellular signaling. 70 human Dbl-family members. |
| PF00905 | Transpeptidase | Penicillin-binding protein (PBP) transpeptidase domain. Catalyzes peptidoglycan cross-linking in bacterial cell wall synthesis. Target of beta-lactam antibiotics (penicillin, cephalosporins). Essential for bacterial cell division and shape maintenance. |
| PF00912 | Transglycosylase | Transglycosylase domain of bifunctional PBPs. Polymerizes NAG-NAM disaccharide units into linear glycan strands of peptidoglycan. Works in concert with the transpeptidase domain (PF00905) for complete cell wall assembly. |
| PF01225 | Mur_ligase_C | Murein ligase catalytic domain (C-terminal). Part of the MurC-F enzyme family that sequentially adds amino acids to UDP-MurNAc during peptidoglycan precursor biosynthesis. Essential for bacterial cell wall construction. |
| PF02861 | Clp_N | ClpB/Hsp100 N-terminal domain. Substrate-binding domain with a hydrophobic groove that recognizes exposed regions of aggregated/misfolded proteins. Primes substrates for ATP-dependent unfolding and translocation through the ClpB central pore. Regulatory role in blocking the translocation channel before substrate engagement. |
| PF02875 | Mur_ligase_M | Murein ligase middle domain. Central structural domain of MurC-F ligases; positions substrates for peptide bond formation during peptidoglycan precursor synthesis. |
| PF07714 | Pkinase_Tyr | Protein tyrosine kinase catalytic domain. Phosphorylates tyrosine residues on target proteins. Core enzymatic activity of receptor tyrosine kinases (RTKs) and non-receptor tyrosine kinases (Src, Abl). Drives cell growth, differentiation, and survival signaling. Frequently co-occurs with SH2 and SH3 domains. |
| PF07724 | AAA_2 | Second AAA ATPase domain variant. Detects additional AAA family members not captured by PF00004. Found in ClpB, ClpA, and related disaggregases/unfoldases. Same ring-shaped hexameric architecture with Walker A/B motifs. |
| PF08245 | Mur_ligase_M | Murein ligase middle domain (alternative Pfam entry). Part of the three-domain architecture of Mur pathway enzymes (N-terminal / middle / C-terminal) involved in peptidoglycan biosynthesis. |
| PF10431 | ClpB_D2-small | C-terminal small domain of ClpB/Hsp100 D2 AAA+ module. Mixed alpha/beta fold essential for hexamer oligomerization. Forms a tight interface with the D2-large domain of neighboring subunits, stabilizing the functional disaggregase ring assembly. |
| PF17871 | AAA_lid_3 | AAA+ lid domain variant 3. Alpha-helical subdomain that caps the nucleotide-binding pocket in AAA+ ATPases. Contributes to nucleotide sensing and allosteric communication between subunits in the hexameric ring. |

---

## GO Terms - Molecular Function

Gene Ontology terms describing the biochemical activities of gene products.

| ID | Name | Description |
|----|------|-------------|
| GO:0000156 | Phosphorelay response regulator activity | Signal transduction activity in two-component systems; receives phosphoryl group and activates downstream response. |
| GO:0000287 | Magnesium ion binding | Binding to Mg2+ ions. Essential cofactor for many enzymes including kinases and nucleases. |
| GO:0000976 | Transcription cis-regulatory region binding | Binding to DNA regions that regulate transcription in cis (same chromosome). |
| GO:0003677 | DNA binding | Selective, non-covalent interaction with DNA. Found in transcription factors, helicases, repair enzymes. |
| GO:0003678 | DNA helicase activity | Unwinding of a DNA helix, driven by ATP hydrolysis. |
| GO:0003724 | RNA helicase activity | Unwinding of an RNA helix, driven by ATP hydrolysis. Essential for splicing, ribosome biogenesis, and translation. |
| GO:0003725 | Double-stranded RNA binding | Binding to double-stranded RNA molecules. Important in RNA interference, viral defense, and RNA processing. |
| GO:0003779 | Actin binding | Interacting selectively with monomeric or filamentous actin. Cytoskeletal regulation and cell motility. |
| GO:0004129 | Cytochrome-c oxidase activity | Catalysis of electron transfer from cytochrome c to oxygen (Complex IV of respiratory chain). |
| GO:0004497 | Monooxygenase activity | Catalysis incorporating one atom of oxygen from O2 into a substrate. Includes cytochrome P450s. |
| GO:0004930 | G protein-coupled receptor activity | Transmembrane signaling receptor activity that transmits signal via G protein activation. |
| GO:0005524 | ATP binding | Binding to adenosine 5'-triphosphate, a universally important coenzyme and enzyme regulator. |
| GO:0008121 | Quinol-cytochrome-c reductase activity | Catalysis of electron transfer from ubiquinol to cytochrome c (Complex III). |
| GO:0008233 | Peptidase activity | Catalysis of the hydrolysis of peptide bonds. Proteolytic enzyme activity. |
| GO:0008658 | Penicillin binding | Binding to penicillin and beta-lactam antibiotics. Characteristic of PBP transpeptidases. |
| GO:0008955 | Peptidoglycan glycosyltransferase activity | Transfer of glycosyl groups during peptidoglycan polymer synthesis. |
| GO:0009002 | Serine-type D-Ala-D-Ala carboxypeptidase activity | Peptidase activity cleaving the terminal D-Ala from peptidoglycan pentapeptide. |
| GO:0016491 | Oxidoreductase activity | Catalysis of an oxidation-reduction reaction. Broad class including dehydrogenases and oxidases. |
| GO:0016787 | Hydrolase activity | Catalysis of the hydrolysis of various bonds. Includes esterases, glycosidases, proteases. |
| GO:0016887 | ATP hydrolysis activity | Catalysis of ATP + H2O = ADP + phosphate. Energy source for molecular motors and transport. |
| GO:0016984 | Ribulose-bisphosphate carboxylase activity | RuBisCO activity; carbon fixation in the Calvin cycle. |
| GO:0020037 | Heme binding | Binding to heme, an iron porphyrin. Found in cytochromes, hemoglobin, and oxidases. |
| GO:0030145 | Manganese ion binding | Binding to Mn2+ ions. Cofactor for enzymes including superoxide dismutase. |
| GO:0043138 | 3'-5' DNA helicase activity | Unwinding a DNA helix in the 3' to 5' direction, driven by ATP hydrolysis. |
| GO:0046872 | Metal ion binding | Binding to any metal ion. Essential for structural zinc fingers, catalytic iron-sulfur clusters, etc. |

---

## GO Terms - Biological Process

Gene Ontology terms describing the larger biological objectives accomplished by gene products.

| ID | Name | Description |
|----|------|-------------|
| GO:0006122 | Mitochondrial electron transport, ubiquinol to cytochrome c | Transfer of electrons from ubiquinol to cytochrome c via Complex III in the mitochondrial respiratory chain. |
| GO:0006123 | Mitochondrial electron transport, cytochrome c to oxygen | Terminal step of oxidative phosphorylation; electrons transferred from cytochrome c to O2 via Complex IV. |
| GO:0006260 | DNA replication | Duplication of DNA molecules; begins at origins of replication, ends with topological separation of copies. |
| GO:0006281 | DNA repair | Restoration of DNA after damage from UV, ionizing radiation, chemical mutagens, or replication errors. Includes base excision, nucleotide excision, mismatch repair, and double-strand break repair. |
| GO:0006310 | DNA recombination | Formation of new genotypes by reassortment of genes. Includes crossing over, homologous recombination, and site-specific recombination. |
| GO:0006353 | DNA-templated transcription termination | Completion of transcription: RNA polymerase pauses, RNA-DNA hybrid dissociates, polymerase releases from template. |
| GO:0006355 | Regulation of DNA-templated transcription | Modulation of the frequency, rate, or extent of transcription from a DNA template. |
| GO:0006397 | mRNA processing | Conversion of primary mRNA transcript into mature mRNA(s) prior to translation. Includes capping, splicing, and polyadenylation. |
| GO:0006417 | Regulation of translation | Modulation of the frequency, rate, or extent of protein synthesis from mRNA or circRNA. |
| GO:0006508 | Proteolysis | Hydrolysis of proteins into smaller polypeptides and/or amino acids by cleavage of peptide bonds. |
| GO:0006915 | Apoptotic process | Programmed cell death proceeding through signaling pathway and execution phases. Characterized by cell rounding, chromatin condensation, nuclear fragmentation, and formation of apoptotic bodies. |
| GO:0006979 | Response to oxidative stress | Cellular response to reactive oxygen species (ROS) and oxidative damage. |
| GO:0007155 | Cell adhesion | Attachment of a cell to another cell or to the extracellular matrix via adhesion molecules. |
| GO:0007399 | Nervous system development | Progression of the nervous system from formation to mature structure, including neurogenesis and axon guidance. |
| GO:0008360 | Regulation of cell shape | Modulation of the surface configuration of a cell. Critical for bacterial morphology (rod vs coccus). |
| GO:0008380 | RNA splicing | Removal of intron sequences from primary RNA transcript and joining of remaining exon sequences. |
| GO:0009252 | Peptidoglycan biosynthetic process | Chemical reactions and pathways forming peptidoglycan, the major structural polymer of bacterial cell walls. |
| GO:0009432 | SOS response | Bacterial DNA damage response: error-prone repair, cell division inhibition, and survival gene induction. |
| GO:0009853 | Photorespiration | Oxygenase activity of RuBisCO followed by salvage of 2-phosphoglycolate; energy-consuming process in C3 plants. |
| GO:0010468 | Regulation of gene expression | Modulation of the frequency, rate, or extent of gene expression at any level (transcription through protein stability). |
| GO:0015990 | Electron transport coupled proton transport | Movement of protons across a membrane driven by electron transport chain activity. Powers ATP synthase. |
| GO:0019253 | Reductive pentose-phosphate cycle | Calvin cycle: carbon fixation pathway using ATP and NADPH to convert CO2 into organic molecules. |
| GO:0030154 | Cell differentiation | Process by which a relatively unspecialized cell acquires specialized structural and functional features. |
| GO:0034605 | Cellular response to heat | Changes in cellular state or activity in response to elevated temperature. Heat shock response. |
| GO:0035556 | Intracellular signal transduction | Signal propagation to downstream components within the cell, triggering changes in function or state. |
| GO:0045087 | Innate immune response | Defense responses mediated by germline-encoded components that directly recognize pathogen components. |
| GO:0045944 | Positive regulation of transcription by RNA polymerase II | Activation or increase of transcription from an RNA polymerase II promoter. |
| GO:0051301 | Cell division | Division and partitioning of cellular components to form more cells. Includes binary fission and mitosis. |
| GO:0051607 | Defense response to virus | Reactions triggered by viral presence that protect the cell or organism. Includes interferon signaling. |
| GO:0071555 | Cell wall organization | Assembly, arrangement, or disassembly of the cell wall in plants, fungi, and prokaryotes. |

---

## GO Terms - Cellular Component

Gene Ontology terms describing the locations where gene products are active.

| ID | Name | Description |
|----|------|-------------|
| GO:0000139 | Golgi membrane | The lipid bilayer surrounding the Golgi apparatus, site of protein modification and sorting. |
| GO:0005634 | Nucleus | Membrane-bounded organelle housing chromosomes; site of DNA replication, RNA synthesis, and processing. |
| GO:0005654 | Nucleoplasm | Nuclear content other than chromosomes or nucleolus. Contains transcription and splicing machinery. |
| GO:0005730 | Nucleolus | Dense nuclear body; site of rRNA transcription, processing, and ribosome assembly. Not membrane-bounded. |
| GO:0005737 | Cytoplasm | Cell contents excluding plasma membrane and nucleus, including organelles and cytosol. |
| GO:0005739 | Mitochondrion | Double-membrane organelle; site of oxidative phosphorylation, TCA cycle, and apoptosis regulation. |
| GO:0005743 | Mitochondrial inner membrane | Inner of the two mitochondrial membranes; location of the electron transport chain and ATP synthase. |
| GO:0005769 | Early endosome | Membrane-bounded compartment receiving material from primary endocytic vesicles. Sorting station. |
| GO:0005789 | Endoplasmic reticulum membrane | Membrane of the ER; site of lipid synthesis and protein translocation/folding. |
| GO:0005813 | Centrosome | Microtubule-organizing center containing centrioles; organizes spindle apparatus during cell division. |
| GO:0005829 | Cytosol | Aqueous part of the cytoplasm, excluding organelles and cytoskeletal structures. |
| GO:0005856 | Cytoskeleton | Network of protein filaments (actin, microtubules, intermediate filaments) providing structural support and motility. |
| GO:0005886 | Plasma membrane | Phospholipid bilayer separating the cell from its external environment. |
| GO:0009507 | Chloroplast | Double-membrane organelle in plants and algae; site of photosynthesis and carbon fixation. |
| GO:0016607 | Nuclear speck | Subnuclear structures enriched in pre-mRNA splicing factors. Storage/assembly sites for splicing machinery. |
| GO:0030288 | Outer membrane-bounded periplasmic space | Space between inner and outer membranes in Gram-negative bacteria. Contains peptidoglycan. |
| GO:0030424 | Axon | Long neuronal process that conducts electrical impulses away from the cell body. |
| GO:0030425 | Dendrite | Branched neuronal projection that receives synaptic input from other neurons. |
| GO:0032993 | Protein-DNA complex | Macromolecular complex of protein and DNA. Includes nucleosomes and transcription factor complexes. |
| GO:0043590 | Bacterial nucleoid | Region within a bacterial cell containing the chromosome(s). Not membrane-bounded. |
| GO:0045275 | Respiratory chain complex III | Cytochrome bc1 complex; transfers electrons from ubiquinol to cytochrome c. |
| GO:0048471 | Perinuclear region of cytoplasm | Cytoplasmic region immediately surrounding the nucleus. Contains Golgi and centrosome. |
| GO:1990904 | Ribonucleoprotein complex | Macromolecular complex containing both RNA and protein. Includes ribosomes, spliceosomes, and snRNPs. |

---

*Generated from et-miner AlphaFold 214M protein mining results (2.84M frequent itemsets, K=1-19). Term definitions sourced from InterPro/Pfam and Gene Ontology (QuickGO/AmiGO).*
