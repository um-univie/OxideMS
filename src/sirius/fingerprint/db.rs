//! Tiered Structural Database Subsystem for CSI:FingerID.
//!
//! Provides fast formula-indexed candidate structure retrieval with multi-tier scoping:
//! - `BioMetabolites`: Curated biological metabolites & natural products (HMDB, ChEBI, COCONUT, MoNA)
//! - `PubChemBio`: PubMed / literature-referenced PubChem molecules
//! - `AllPubChem`: Exhaustive 124.6M molecule PubChem search (on disk via `CID-SMILES`)

use super::dictionary::{BinaryFingerprint, FingerprintDictionary, MorganVocabulary};
use super::predictor::PredictedFingerprint;
use super::search::FingerprintSearcher;
use crate::sirius::formula::Formula;
use crate::sirius::workflow::CandidateStructureMatch;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use rayon::prelude::*;

/// Biological origin or sample context for intelligent candidate structure filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleContext {
    /// General biological metabolomics (all curated biological metabolites).
    AllBio,
    /// Human clinical metabolomics (HMDB human endogenous, blood, urine, CSF, clinical drugs).
    Human,
    /// Plant / Botanical metabolomics (phytochemicals, flavonoids, plant secondary metabolites).
    Plant,
    /// Microbial / Bacterial / Fungal natural products (antibiotics, polyketides, NRPs).
    Microbial,
    /// Marine natural products (marine sponges, algae, marine metabolites).
    Marine,
    /// Lipidomics (LIPID MAPS, fatty acyls, glycerolipids, sphingolipids, sterols).
    Lipids,
    /// Food and dietary metabolomics (food-derived metabolites).
    Food,
    /// Unconstrained chemical search across all known molecules.
    AllChemicals,
}

impl Default for SampleContext {
    fn default() -> Self {
        Self::AllBio
    }
}

/// Database scope / tier for structural candidate filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatabaseScope {
    /// High-precision biological metabolites and natural products (HMDB, ChEBI, COCONUT, MoNA).
    BioMetabolites,
    /// PubChem compounds with at least one scientific literature citation or biological assay.
    PubChemBio,
    /// Exhaustive search across all 124.6 million PubChem compounds.
    AllPubChem,
    /// Custom user-provided candidate pool.
    Custom,
}

/// A candidate molecular structure stored in the database.
#[derive(Debug, Clone, PartialEq)]
pub struct DatabaseStructure {
    /// Unique identifier (e.g. "CID:631170", "HMDB0000001", "MoNA:12345")
    pub id: String,
    /// Common or IUPAC chemical name
    pub name: String,
    /// Canonical SMILES representation
    pub smiles: String,
    /// Optional Standard InChIKey
    pub inchi_key: Option<String>,
    /// Elemental molecular formula
    pub formula: Formula,
    /// Precomputed 2,214-bit binary fingerprint (166 MACCS + 2048 Morgan ECFP4)
    pub fingerprint: BinaryFingerprint,
    /// Biological origin tags (e.g. Human, Plant, Microbial, Lipids)
    pub contexts: Vec<SampleContext>,
}

/// In-memory formula-indexed structural candidate database with optional disk fallback.
#[derive(Debug, Clone)]
pub struct StructureDatabase {
    /// Database tier scope
    pub scope: DatabaseScope,
    /// Fast in-memory candidate index grouped by exact elemental formula
    pub entries_by_formula: HashMap<Formula, Vec<DatabaseStructure>>,
    /// Total count of unique structures in memory
    pub total_structures: usize,
    /// Optional path to full disk-backed `CID-SMILES` for exhaustive fallback queries
    pub pubchem_fallback_path: Option<PathBuf>,
    /// The fingerprint layout every stored fingerprint was computed under.
    ///
    /// Candidate fingerprints only mean anything against a prediction made in the same
    /// layout, and the Morgan vocabulary is part of that layout. Carrying it here is
    /// what lets [`query_and_rank`](Self::query_and_rank) refuse a mismatch instead of
    /// returning a confident ranking of nonsense.
    pub dictionary: FingerprintDictionary,
}

impl Default for StructureDatabase {
    fn default() -> Self {
        Self::new(DatabaseScope::BioMetabolites)
    }
}

impl StructureDatabase {
    /// Creates a new empty structural database with the specified tier scope.
    pub fn new(scope: DatabaseScope) -> Self {
        Self {
            scope,
            entries_by_formula: HashMap::new(),
            total_structures: 0,
            pubchem_fallback_path: None,
            dictionary: FingerprintDictionary::new(),
        }
    }

    /// Sets the fingerprint layout structures are indexed under.
    ///
    /// Must be done before ingesting anything: it decides what the stored fingerprints
    /// mean. Take it from the model that will query this database, via
    /// [`FingerprintSearcher::for_predictor`](super::search::FingerprintSearcher::for_predictor).
    pub fn with_dictionary(mut self, dictionary: FingerprintDictionary) -> Self {
        assert!(
            self.total_structures == 0,
            "the fingerprint layout must be chosen before structures are ingested"
        );
        self.dictionary = dictionary;
        self
    }

    /// Sets the path to the full PubChem `CID-SMILES` file for fallback searches.
    pub fn with_pubchem_fallback<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.pubchem_fallback_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Inserts a new structure entry into the in-memory database index.
    pub fn insert(&mut self, structure: DatabaseStructure) {
        let entry = self
            .entries_by_formula
            .entry(structure.formula)
            .or_insert_with(Vec::new);

        // Deduplicate by SMILES
        if !entry.iter().any(|s| s.smiles == structure.smiles) {
            entry.push(structure);
            self.total_structures += 1;
        }
    }

    /// Retrieves candidate structures matching an exact molecular formula filtered by sample context.
    pub fn get_candidates_for_context(&self, formula: &Formula, context: SampleContext) -> Vec<DatabaseStructure> {
        let raw_candidates = self.get_candidates(formula);
        if context == SampleContext::AllBio || context == SampleContext::AllChemicals {
            return raw_candidates;
        }

        let filtered: Vec<DatabaseStructure> = raw_candidates
            .into_iter()
            .filter(|cand| {
                cand.contexts.is_empty()
                    || cand.contexts.contains(&context)
                    || cand.contexts.contains(&SampleContext::AllBio)
            })
            .collect();

        filtered
    }

    /// Retrieves all candidate structural isomers matching an exact molecular formula.
    ///
    /// If in-memory candidates exist, returns them immediately (< 1µs).
    /// If in-memory candidates are empty and `pubchem_fallback_path` is configured,
    /// scans PubChem on disk, computes fingerprints, caches into memory, and returns them.
    pub fn get_candidates(&self, formula: &Formula) -> Vec<DatabaseStructure> {
        if let Some(candidates) = self.entries_by_formula.get(formula) {
            if !candidates.is_empty() {
                return candidates.clone();
            }
        }

        // Optional on-demand PubChem fallback scan if formula is missing in RAM
        if let Some(ref path) = self.pubchem_fallback_path {
            return self.scan_pubchem_for_formula(path, formula);
        }

        Vec::new()
    }

    /// Scores all candidate structures for a given formula against a predicted fingerprint,
    /// returning the ranked top-k structural matches.
    pub fn query_and_rank(
        &self,
        formula: &Formula,
        predicted_fp: &PredictedFingerprint,
        searcher: &FingerprintSearcher,
        context: SampleContext,
        top_k: usize,
    ) -> Vec<CandidateStructureMatch> {
        // A fingerprint means nothing outside the layout it was computed in. Scoring a
        // database indexed under one Morgan vocabulary against a model trained under
        // another produces a perfectly confident ranking of noise, and nothing
        // downstream could tell. Fail loudly instead.
        assert_eq!(
            self.dictionary.morgan_vocabulary,
            searcher.dictionary().morgan_vocabulary,
            "structure database and predictor disagree on the Morgan vocabulary: \
             the database was indexed under a different fingerprint layout than the \
             model predicts in, so re-index it with the model's dictionary"
        );

        let owned;
        let source = if let Some(entries) = self.entries_by_formula.get(formula).filter(|entries| !entries.is_empty()) {
            entries.as_slice()
        } else {
            owned = self.get_candidates(formula);
            owned.as_slice()
        };
        let candidates: Vec<_> = source.iter().filter(|cand| {
            context == SampleContext::AllBio || context == SampleContext::AllChemicals
                || cand.contexts.is_empty() || cand.contexts.contains(&context)
                || cand.contexts.contains(&SampleContext::AllBio)
        }).collect();
        if candidates.is_empty() {
            return Vec::new();
        }

        let table = searcher.prepare(predicted_fp);

        let mut ranked: Vec<_> = candidates.iter().enumerate()
            .map(|(i, cand)| (i, table.score(&cand.fingerprint))).collect();
        // Preserve input-order tie breaking without cloning candidate metadata.
        let compare = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if ranked.len() > top_k {
            ranked.select_nth_unstable_by(top_k, compare);
            ranked.truncate(top_k);
        }
        ranked.sort_by(compare);
        ranked.into_iter().map(|(i, score)| {
            let cand = &candidates[i];
            CandidateStructureMatch {
                name: cand.name.clone(), smiles: cand.smiles.clone(), inchi_key: cand.inchi_key.clone(),
                tanimoto_similarity: FingerprintSearcher::tanimoto_similarity(predicted_fp, &cand.fingerprint, 0.5) as f32,
                mod_platt_score: score,
            }
        }).collect()

    }

    /// Ingests all reference compounds and authentic standards from a MoNA / MassBank MSP file in parallel with Rayon.
    pub fn ingest_mona_msp<P: AsRef<Path>>(&mut self, path: P, max_records: usize) -> io::Result<usize> {
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

        let mut raw_records = Vec::new();
        let mut current_name = String::new();
        let mut current_formula = String::new();
        let mut current_smiles = String::new();
        let mut current_inchikey = None;
        let mut count = 0usize;

        for line_res in reader.lines() {
            if count >= max_records {
                break;
            }
            let line = match line_res {
                Ok(l) => l,
                Err(_) => continue,
            };
            let trimmed = line.trim();

            if trimmed.is_empty() {
                if !current_smiles.is_empty() && !current_formula.is_empty() {
                    raw_records.push((
                        if !current_name.is_empty() {
                            std::mem::take(&mut current_name)
                        } else {
                            "Unknown".to_string()
                        },
                        std::mem::take(&mut current_formula),
                        std::mem::take(&mut current_smiles),
                        current_inchikey.take(),
                    ));
                    count += 1;
                }
                current_name.clear();
                current_formula.clear();
                current_smiles.clear();
                current_inchikey = None;
                continue;
            }

            // Quick skip of numeric peak data lines
            if let Some(first_byte) = trimmed.as_bytes().first() {
                if first_byte.is_ascii_digit() {
                    continue;
                }
            }

            if let Some(val) = trimmed.strip_prefix("Name: ") {
                current_name = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("NAME: ") {
                current_name = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("Formula: ") {
                current_formula = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("FORMULA: ") {
                current_formula = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("SMILES: ") {
                current_smiles = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("InChIKey: ") {
                current_inchikey = Some(val.trim().to_string());
            } else if let Some(val) = trimmed.strip_prefix("INCHIKEY: ") {
                current_inchikey = Some(val.trim().to_string());
            } else if trimmed.starts_with("Comments: ") || trimmed.starts_with("Comment: ") {
                for token in trimmed.split('"') {
                    if token.starts_with("SMILES=") {
                        current_smiles = token["SMILES=".len()..].trim().to_string();
                    } else if token.starts_with("formula=") {
                        current_formula = token["formula=".len()..].trim().to_string();
                    } else if token.starts_with("inchikey=") {
                        current_inchikey = Some(token["inchikey=".len()..].trim().to_string());
                    }
                }
            }
        }

        // Parallel fingerprint computation across all extracted records
        let parsed_structures: Vec<DatabaseStructure> = raw_records
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, (name, formula_str, smiles, inchikey))| {
                let formula = Formula::from_str(&formula_str).ok()?;
                let fp_bits = self.dictionary.ground_truth_fingerprint(&smiles)?;
                let fp = BinaryFingerprint::from_bits(fp_bits);

                Some(DatabaseStructure {
                    id: format!("MoNA:{}", idx + 1),
                    name,
                    smiles,
                    inchi_key: inchikey,
                    formula,
                    fingerprint: fp,
                    contexts: vec![SampleContext::AllBio],
                })
            })
            .collect();

        let inserted_count = parsed_structures.len();
        for s in parsed_structures {
            self.insert(s);
        }

        Ok(inserted_count)
    }

    /// Ingests candidate structures from a tab-separated `<ID>\t<SMILES>` file in parallel using Rayon.
    pub fn ingest_smiles_tsv<P: AsRef<Path>>(
        &mut self,
        path: P,
        max_records: usize,
        context: SampleContext,
    ) -> io::Result<usize> {
        // Cloned rather than borrowed: ingestion inserts into `self` while the parallel
        // closures below are still reading the layout.
        let dictionary = self.dictionary.clone();
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

        let mut lines_chunk = Vec::with_capacity(10000);
        let mut total_processed = 0usize;

        for line_res in reader.lines() {
            if total_processed >= max_records {
                break;
            }
            if let Ok(line) = line_res {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    lines_chunk.push(trimmed);

                    if lines_chunk.len() >= 10000 || (total_processed + lines_chunk.len()) >= max_records {
                        let parsed: Vec<DatabaseStructure> = lines_chunk
                            .par_iter()
                            .filter_map(|l| {
                                if let Some(tab_idx) = l.find('\t') {
                                    let id_str = &l[..tab_idx];
                                    let smiles = &l[tab_idx + 1..];

                                    if let Some(fp_bits) = dictionary.ground_truth_fingerprint(smiles) {
                                        if let Ok(mol) = oxidane::molecule::molecular_graph::MolecularGraph::from_smiles_single(smiles) {
                                            let elements = &mol.atomic_numbers;
                                            let mut c = 0;
                                            let mut h = 0;
                                            let mut n = 0;
                                            let mut o = 0;
                                            let mut p = 0;
                                            let mut s = 0;
                                            for &at in elements {
                                                match at {
                                                    6 => c += 1,
                                                    1 => h += 1,
                                                    7 => n += 1,
                                                    8 => o += 1,
                                                    15 => p += 1,
                                                    16 => s += 1,
                                                    _ => {}
                                                }
                                            }
                                            if c > 0 || n > 0 || o > 0 {
                                                let formula = Formula::chnops(c, h, n, o, p, s);
                                                let fp = BinaryFingerprint::from_bits(fp_bits);

                                                return Some(DatabaseStructure {
                                                    id: format!("CID:{}", id_str),
                                                    name: format!("PubChem CID {}", id_str),
                                                    smiles: smiles.to_string(),
                                                    inchi_key: None,
                                                    formula,
                                                    fingerprint: fp,
                                                    contexts: vec![context],
                                                });
                                            }
                                        }
                                    }
                                }
                                None
                            })
                            .collect();

                        for s in parsed {
                            self.insert(s);
                            total_processed += 1;
                        }
                        lines_chunk.clear();
                    }
                }
            }
        }

        Ok(total_processed)
    }

    /// Ingests candidate structures from standard CSV databases (e.g. HMDB, COCONUT, MetFrag) in parallel with Rayon.
    pub fn ingest_structure_csv<P: AsRef<Path>>(
        &mut self,
        path: P,
        max_records: usize,
        default_context: SampleContext,
    ) -> io::Result<usize> {
        // Cloned rather than borrowed: ingestion inserts into `self` while the parallel
        // closures below are still reading the layout.
        let dictionary = self.dictionary.clone();
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);

        let mut header_line = String::new();
        if reader.read_line(&mut header_line)? == 0 {
            return Ok(0);
        }

        let header_tokens: Vec<String> = header_line
            .trim()
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_lowercase())
            .collect();

        let mut id_col = None;
        let mut name_col = None;
        let mut formula_col = None;
        let mut smiles_col = None;
        let mut inchikey_col = None;

        for (idx, h) in header_tokens.iter().enumerate() {
            if h.contains("identifier") || h.contains("accession") || h == "id" {
                id_col = Some(idx);
            } else if h.contains("compoundname") || h.contains("name") {
                if name_col.is_none() {
                    name_col = Some(idx);
                }
            } else if h.contains("formula") {
                formula_col = Some(idx);
            } else if h.contains("smiles") {
                smiles_col = Some(idx);
            } else if h.contains("inchikey") {
                inchikey_col = Some(idx);
            }
        }

        let smiles_idx = match smiles_col {
            Some(idx) => idx,
            None => return Err(io::Error::new(io::ErrorKind::InvalidData, "No SMILES column found in CSV")),
        };

        let mut lines_chunk = Vec::with_capacity(10000);
        let mut total_processed = 0usize;

        for line_res in reader.lines() {
            if total_processed >= max_records {
                break;
            }
            if let Ok(line) = line_res {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    lines_chunk.push(trimmed);

                    if lines_chunk.len() >= 10000 || (total_processed + lines_chunk.len()) >= max_records {
                        let parsed: Vec<DatabaseStructure> = lines_chunk
                            .par_iter()
                            .filter_map(|l| {
                                let mut tokens = Vec::new();
                                let mut in_quote = false;
                                let mut cur_tok = String::new();
                                for c in l.chars() {
                                    if c == '"' {
                                        in_quote = !in_quote;
                                    } else if c == ',' && !in_quote {
                                        tokens.push(std::mem::take(&mut cur_tok));
                                    } else {
                                        cur_tok.push(c);
                                    }
                                }
                                tokens.push(cur_tok);

                                if tokens.len() <= smiles_idx {
                                    return None;
                                }

                                let smiles = tokens[smiles_idx].trim().trim_matches('"');
                                if smiles.is_empty() {
                                    return None;
                                }

                                let id = id_col
                                    .and_then(|idx| tokens.get(idx))
                                    .map(|s| s.trim().trim_matches('"').to_string())
                                    .unwrap_or_else(|| "Unknown".to_string());

                                let name = name_col
                                    .and_then(|idx| tokens.get(idx))
                                    .map(|s| s.trim().trim_matches('"').to_string())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or_else(|| id.clone());

                                let formula_opt = formula_col
                                    .and_then(|idx| tokens.get(idx))
                                    .and_then(|f_str| Formula::from_str(f_str.trim().trim_matches('"')).ok());

                                let inchi_key = inchikey_col
                                    .and_then(|idx| tokens.get(idx))
                                    .map(|s| s.trim().trim_matches('"').to_string())
                                    .filter(|s| !s.is_empty());

                                if let Some(fp_bits) = dictionary.ground_truth_fingerprint(smiles) {
                                    let formula = if let Some(f) = formula_opt {
                                        f
                                    } else if let Ok(mol) = oxidane::molecule::molecular_graph::MolecularGraph::from_smiles_single(smiles) {
                                        let elements = &mol.atomic_numbers;
                                        let mut c = 0;
                                        let mut h = 0;
                                        let mut n = 0;
                                        let mut o = 0;
                                        let mut p = 0;
                                        let mut s = 0;
                                        for &at in elements {
                                            match at {
                                                6 => c += 1,
                                                1 => h += 1,
                                                7 => n += 1,
                                                8 => o += 1,
                                                15 => p += 1,
                                                16 => s += 1,
                                                _ => {}
                                            }
                                        }
                                        Formula::chnops(c, h, n, o, p, s)
                                    } else {
                                        return None;
                                    };

                                    let fp = BinaryFingerprint::from_bits(fp_bits);
                                    return Some(DatabaseStructure {
                                        id,
                                        name,
                                        smiles: smiles.to_string(),
                                        inchi_key,
                                        formula,
                                        fingerprint: fp,
                                        contexts: vec![default_context, SampleContext::AllBio],
                                    });
                                }
                                None
                            })
                            .collect();

                        for s in parsed {
                            self.insert(s);
                            total_processed += 1;
                        }
                        lines_chunk.clear();
                    }
                }
            }
        }

        Ok(total_processed)
    }

    /// Fast scan of PubChem CID-SMILES for an exact formula on demand.
    fn scan_pubchem_for_formula(&self, path: &Path, formula: &Formula) -> Vec<DatabaseStructure> {
        // Bound out of the loop so the parallel closures can see the layout.
        let dictionary = &self.dictionary;
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::with_capacity(4 * 1024 * 1024, file);

        let mut matches = Vec::new();
        let target_c = formula.c() as usize;
        let target_o = formula.o() as usize;
        let target_n = formula.n() as usize;
        let mut scanned_lines = 0usize;
        let max_scan_lines = 1_000_000; // Cap live scan for instant responsiveness

        for line in reader.lines() {
            if matches.len() >= 200 || scanned_lines >= max_scan_lines {
                break;
            }
            scanned_lines += 1;

            if let Ok(line_str) = line {
                if let Some(tab_idx) = line_str.find('\t') {
                    let cid_str = &line_str[..tab_idx];
                    let smiles = &line_str[tab_idx + 1..];

                    // Quick character count filter before expensive parsing
                    let c_count = smiles.chars().filter(|&c| c == 'C' || c == 'c').count();
                    let o_count = smiles.chars().filter(|&c| c == 'O' || c == 'o').count();
                    let n_count = smiles.chars().filter(|&c| c == 'N' || c == 'n').count();

                    if c_count == target_c && o_count == target_o && n_count == target_n {
                        if let Some(fp_bits) = dictionary.ground_truth_fingerprint(smiles) {
                            let fp = BinaryFingerprint::from_bits(fp_bits);
                            matches.push(DatabaseStructure {
                                id: format!("CID:{}", cid_str),
                                name: format!("PubChem CID {}", cid_str),
                                smiles: smiles.to_string(),
                                inchi_key: None,
                                formula: *formula,
                                fingerprint: fp,
                                contexts: vec![SampleContext::AllChemicals],
                            });
                        }
                    }
                }
            }
        }

        matches
    }

    /// Magic bytes for OxideMS Structure Database checkpoint files.
    pub const DB_MAGIC: &'static [u8; 4] = b"OXDB";
    /// Checkpoint file version.
    /// - `1`: structures only.
    /// - `2`: same, plus the Morgan vocabulary the fingerprints were computed under.
    pub const DB_VERSION: u32 = 2;
    const DB_VERSION_NO_VOCABULARY: u32 = 1;

    /// Serializes the structure database to a compact binary `.oxdb` file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);

        writer.write_all(Self::DB_MAGIC)?;
        writer.write_all(&Self::DB_VERSION.to_le_bytes())?;

        let scope_byte = match self.scope {
            DatabaseScope::BioMetabolites => 0u8,
            DatabaseScope::PubChemBio => 1u8,
            DatabaseScope::AllPubChem => 2u8,
            DatabaseScope::Custom => 3u8,
        };
        writer.write_all(&[scope_byte])?;

        let vocabulary = self.dictionary.morgan_vocabulary.ids();
        writer.write_all(&(vocabulary.len() as u32).to_le_bytes())?;
        for &id in vocabulary {
            writer.write_all(&id.to_le_bytes())?;
        }

        writer.write_all(&(self.total_structures as u64).to_le_bytes())?;
        writer.write_all(&(self.entries_by_formula.len() as u32).to_le_bytes())?;

        for (formula, structures) in &self.entries_by_formula {
            let formula_str = formula.to_string();
            let formula_bytes = formula_str.as_bytes();
            writer.write_all(&(formula_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(formula_bytes)?;

            writer.write_all(&(structures.len() as u32).to_le_bytes())?;

            for s in structures {
                let id_bytes = s.id.as_bytes();
                writer.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
                writer.write_all(id_bytes)?;

                let name_bytes = s.name.as_bytes();
                writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
                writer.write_all(name_bytes)?;

                let smiles_bytes = s.smiles.as_bytes();
                writer.write_all(&(smiles_bytes.len() as u16).to_le_bytes())?;
                writer.write_all(smiles_bytes)?;

                if let Some(ref ik) = s.inchi_key {
                    let ik_bytes = ik.as_bytes();
                    writer.write_all(&(ik_bytes.len() as u16).to_le_bytes())?;
                    writer.write_all(ik_bytes)?;
                } else {
                    writer.write_all(&0u16.to_le_bytes())?;
                }

                writer.write_all(&(s.contexts.len() as u8).to_le_bytes())?;
                for ctx in &s.contexts {
                    let c_byte = match ctx {
                        SampleContext::AllBio => 0u8,
                        SampleContext::Human => 1u8,
                        SampleContext::Plant => 2u8,
                        SampleContext::Microbial => 3u8,
                        SampleContext::Marine => 4u8,
                        SampleContext::Lipids => 5u8,
                        SampleContext::Food => 6u8,
                        SampleContext::AllChemicals => 7u8,
                    };
                    writer.write_all(&[c_byte])?;
                }

                let packed_fp = s.fingerprint.to_packed_bytes();
                writer.write_all(&packed_fp)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Loads a pre-indexed structure database from a compact binary `.oxdb` file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != Self::DB_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid structure database magic bytes",
            ));
        }

        let mut ver_bytes = [0u8; 4];
        reader.read_exact(&mut ver_bytes)?;
        let version = u32::from_le_bytes(ver_bytes);
        if version != Self::DB_VERSION && version != Self::DB_VERSION_NO_VOCABULARY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported database version: {}", version),
            ));
        }

        let mut scope_buf = [0u8; 1];
        reader.read_exact(&mut scope_buf)?;
        let scope = match scope_buf[0] {
            0 => DatabaseScope::BioMetabolites,
            1 => DatabaseScope::PubChemBio,
            2 => DatabaseScope::AllPubChem,
            _ => DatabaseScope::Custom,
        };

        // Version 1 files predate the vocabulary, so they are plain folded Morgan.
        let mut vocabulary = Vec::new();
        if version == Self::DB_VERSION {
            let mut count_buf = [0u8; 4];
            reader.read_exact(&mut count_buf)?;
            let count = u32::from_le_bytes(count_buf) as usize;
            let mut id_bytes = vec![0u8; count * 4];
            reader.read_exact(&mut id_bytes)?;
            vocabulary = id_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
        }

        let mut total_struct_buf = [0u8; 8];
        reader.read_exact(&mut total_struct_buf)?;
        let total_structures = u64::from_le_bytes(total_struct_buf) as usize;

        let mut formula_count_buf = [0u8; 4];
        reader.read_exact(&mut formula_count_buf)?;
        let num_formulas = u32::from_le_bytes(formula_count_buf) as usize;

        let mut entries_by_formula = HashMap::with_capacity(num_formulas);

        for _ in 0..num_formulas {
            let mut f_len_buf = [0u8; 2];
            reader.read_exact(&mut f_len_buf)?;
            let f_len = u16::from_le_bytes(f_len_buf) as usize;
            let mut f_str_bytes = vec![0u8; f_len];
            reader.read_exact(&mut f_str_bytes)?;
            let formula_str = String::from_utf8_lossy(&f_str_bytes);
            let formula = Formula::from_str(&formula_str)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{:?}", e)))?;

            let mut struct_count_buf = [0u8; 4];
            reader.read_exact(&mut struct_count_buf)?;
            let num_structures = u32::from_le_bytes(struct_count_buf) as usize;

            let mut structures = Vec::with_capacity(num_structures);

            for _ in 0..num_structures {
                let mut id_len_buf = [0u8; 2];
                reader.read_exact(&mut id_len_buf)?;
                let id_len = u16::from_le_bytes(id_len_buf) as usize;
                let mut id_bytes = vec![0u8; id_len];
                reader.read_exact(&mut id_bytes)?;
                let id = String::from_utf8_lossy(&id_bytes).to_string();

                let mut name_len_buf = [0u8; 2];
                reader.read_exact(&mut name_len_buf)?;
                let name_len = u16::from_le_bytes(name_len_buf) as usize;
                let mut name_bytes = vec![0u8; name_len];
                reader.read_exact(&mut name_bytes)?;
                let name = String::from_utf8_lossy(&name_bytes).to_string();

                let mut smiles_len_buf = [0u8; 2];
                reader.read_exact(&mut smiles_len_buf)?;
                let smiles_len = u16::from_le_bytes(smiles_len_buf) as usize;
                let mut smiles_bytes = vec![0u8; smiles_len];
                reader.read_exact(&mut smiles_bytes)?;
                let smiles = String::from_utf8_lossy(&smiles_bytes).to_string();

                let mut ik_len_buf = [0u8; 2];
                reader.read_exact(&mut ik_len_buf)?;
                let ik_len = u16::from_le_bytes(ik_len_buf) as usize;
                let inchi_key = if ik_len > 0 {
                    let mut ik_bytes = vec![0u8; ik_len];
                    reader.read_exact(&mut ik_bytes)?;
                    Some(String::from_utf8_lossy(&ik_bytes).to_string())
                } else {
                    None
                };

                let mut ctx_len_buf = [0u8; 1];
                reader.read_exact(&mut ctx_len_buf)?;
                let num_ctx = ctx_len_buf[0] as usize;
                let mut contexts = Vec::with_capacity(num_ctx);
                for _ in 0..num_ctx {
                    let mut c_buf = [0u8; 1];
                    reader.read_exact(&mut c_buf)?;
                    let ctx = match c_buf[0] {
                        1 => SampleContext::Human,
                        2 => SampleContext::Plant,
                        3 => SampleContext::Microbial,
                        4 => SampleContext::Marine,
                        5 => SampleContext::Lipids,
                        6 => SampleContext::Food,
                        7 => SampleContext::AllChemicals,
                        _ => SampleContext::AllBio,
                    };
                    contexts.push(ctx);
                }

                let mut packed_fp = [0u8; BinaryFingerprint::PACKED_BYTES];
                reader.read_exact(&mut packed_fp)?;
                let fingerprint = BinaryFingerprint::from_packed_bytes(&packed_fp);

                structures.push(DatabaseStructure {
                    id,
                    name,
                    smiles,
                    inchi_key,
                    formula,
                    fingerprint,
                    contexts,
                });
            }

            entries_by_formula.insert(formula, structures);
        }

        let mut dictionary = FingerprintDictionary::new();
        if !vocabulary.is_empty() {
            dictionary.set_morgan_vocabulary(MorganVocabulary::from_ids(vocabulary));
        }

        Ok(Self {
            scope,
            entries_by_formula,
            total_structures,
            pubchem_fallback_path: None,
            dictionary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_database_insert_and_query() {
        let mut db = StructureDatabase::new(DatabaseScope::BioMetabolites);

        let zapotin_smiles = "COC1=C(C=C(C=C1)C2=CC(=O)C3=C(O2)C(=C(C=C3OC)OC)OC)OC";
        let formula = Formula::from_str("C19H18O6").unwrap();
        let fp_bits = FingerprintDictionary::new().ground_truth_fingerprint(zapotin_smiles).unwrap();
        let fp = BinaryFingerprint::from_bits(fp_bits);

        db.insert(DatabaseStructure {
            id: "CID:631170".to_string(),
            name: "Zapotin".to_string(),
            smiles: zapotin_smiles.to_string(),
            inchi_key: None,
            formula,
            fingerprint: fp,
            contexts: vec![SampleContext::Plant, SampleContext::AllBio],
        });

        let all_cands = db.get_candidates_for_context(&formula, SampleContext::AllBio);
        assert_eq!(all_cands.len(), 1);

        let plant_cands = db.get_candidates_for_context(&formula, SampleContext::Plant);
        assert_eq!(plant_cands.len(), 1);

        let human_cands = db.get_candidates_for_context(&formula, SampleContext::Human);
        assert_eq!(human_cands.len(), 1); // Plant compound tagged with AllBio is included in Human if AllBio
    }

    #[test]
    fn test_structure_database_save_load_roundtrip() {
        let mut db = StructureDatabase::new(DatabaseScope::BioMetabolites);

        let zapotin_smiles = "COC1=C(C=C(C=C1)C2=CC(=O)C3=C(O2)C(=C(C=C3OC)OC)OC)OC";
        let formula = Formula::from_str("C19H18O6").unwrap();
        let fp_bits = FingerprintDictionary::new().ground_truth_fingerprint(zapotin_smiles).unwrap();
        let fp = BinaryFingerprint::from_bits(fp_bits);

        db.insert(DatabaseStructure {
            id: "CID:631170".to_string(),
            name: "Zapotin".to_string(),
            smiles: zapotin_smiles.to_string(),
            inchi_key: Some("CLXVBVLQKLQNRQ-UHFFFAOYSA-N".to_string()),
            formula,
            fingerprint: fp,
            contexts: vec![SampleContext::Plant, SampleContext::AllBio],
        });

        let tmp_path = "/tmp/test_db_roundtrip.oxdb";
        db.save_to_file(tmp_path).expect("Failed to save database");

        let loaded = StructureDatabase::load_from_file(tmp_path).expect("Failed to load database");
        assert_eq!(loaded.total_structures, 1);
        assert_eq!(loaded.scope, DatabaseScope::BioMetabolites);

        let cands = loaded.get_candidates(&formula);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].name, "Zapotin");
        assert_eq!(cands[0].smiles, zapotin_smiles);
        assert_eq!(cands[0].inchi_key.as_deref(), Some("CLXVBVLQKLQNRQ-UHFFFAOYSA-N"));
        assert_eq!(cands[0].contexts, vec![SampleContext::Plant, SampleContext::AllBio]);

        let _ = std::fs::remove_file(tmp_path);
    }
}
