//! This module provides high-performance parsing and writing of Mascot Generic Format (MGF) files.
//!
//! MGF is a plain-text format commonly used in mass spectrometry to store peak lists.
//! This implementation utilizes memory-mapping and parallel processing for maximum throughput.

use crate::clustering::cluster::engine::SpectrumMetadata;
use crate::spectrum::{MSNSpectrum, MassSpectralRecord, Peak};
use crate::workflow::WorkFlowManager;
use rkyv::{Archive, Deserialize, Serialize};
use compact_str::CompactString;
use fast_float::parse;
use memchr::memmem;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::path::Path;
use std::sync::Mutex;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

/// An estimate of the average size (in bytes) of a single MGF spectrum.
/// Used to pre-allocate memory when memory-mapping files.
pub const MGF_SPECTRA_SIZE: usize = 5_000;

/// Charge for a precursor whose charge is missing or zero, from evidence in
/// the spectrum only: 3 or 2 if the matching isotope spacing is found next to
/// the precursor, with the sign given by the polarity. Without such evidence
/// the charge stays 0 (unknown); unknown-charge spectra are clustered only
/// with each other, or removed by [`crate::workflow::DropUnknownCharge`].
pub(crate) fn recover_charge(peaks: &[Peak], precursor_mz: Option<f64>, is_negative: bool) -> i8 {
    let Some(pmz) = precursor_mz else {
        return 0;
    };
    let tol = 0.02;
    let detected_charge = if peaks.iter().any(|p| (p.mz - (pmz + 0.3344)).abs() < tol) {
        3
    } else if peaks.iter().any(|p| (p.mz - (pmz + 0.5017)).abs() < tol) {
        2
    } else {
        return 0;
    };
    if is_negative { -detected_charge } else { detected_charge }
}

/// Parses an MGF `CHARGE` value into `(charge, has_explicit_sign)`.
///
/// Accepts `2`, `2+`, `+2`, `1-`, `-1`, `2.0`, `2.0+` and takes the first
/// entry of lists such as `2+ and 3+`. Returns 0 (unknown) for `0`, a
/// non-integral value or anything unparseable, so the caller can recover it.
pub(crate) fn parse_mgf_charge(val: &str) -> (i8, bool) {
    let first = val
        .split(|c: char| c == ',' || c.is_whitespace())
        .find(|t| !t.is_empty())
        .unwrap_or("");
    let (body, sign) = if let Some(b) = first.strip_suffix('+').or_else(|| first.strip_prefix('+')) {
        (b, Some(1i8))
    } else if let Some(b) = first.strip_suffix('-').or_else(|| first.strip_prefix('-')) {
        (b, Some(-1i8))
    } else {
        (first, None)
    };
    let Ok(value) = body.parse::<f64>() else {
        return (0, false);
    };
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > i8::MAX as f64 {
        return (0, false);
    }
    let magnitude = value.abs() as i8;
    if magnitude == 0 {
        return (0, false);
    }
    match sign {
        Some(s) => (s * magnitude, true),
        None => (magnitude, false),
    }
}

/// Width (Da) of each mass bucket in [`parse_file_low_mem`](MgfParser::parse_file_low_mem)'s
/// shard pool. 25 Da — many multiples of a typical `mz_tolerance` (20
/// ppm ~= 0.03 Da at 1500 Da) so a clustering partition's narrow mass window
/// almost always falls inside a single bucket/shard file, while keeping the
/// pool to ~60 shard files over 50-1500 Da (1 Da buckets needed ~1,450), instead of
/// scattering across every shard the way round-robin-by-parse-order
/// assignment did. The bucket *range* (min/max Da) is not a fixed constant:
/// it's passed in per call, derived from the caller's configured mz range
/// (e.g. `ClusterEngine::hasher`'s `min_mz`/`max_mz` for whichever preset is
/// in use) — spectra outside that range are filtered out before ingestion
/// anyway, so sizing the pool past it would only create dead buckets.
pub const MASS_BUCKET_WIDTH_DA: f64 = 25.0;

/// Number of shards in the mass-bucketed pool for a given `[min_da, max_da)`
/// range (fixed regardless of corpus size or thread count).
fn num_mass_buckets(min_da: f64, max_da: f64) -> usize {
    ((max_da - min_da) / MASS_BUCKET_WIDTH_DA).ceil().max(1.0) as usize
}

/// Maps a neutral mass to its shard index in the mass-bucketed pool, clamping
/// out-of-range (including non-finite) masses into the first/last bucket
/// rather than panicking or growing the pool.
fn mass_bucket_index(neutral_mass: f64, min_da: f64, max_da: f64) -> usize {
    let n = num_mass_buckets(min_da, max_da);
    if !neutral_mass.is_finite() {
        return n - 1;
    }
    let clamped = neutral_mass.clamp(min_da, max_da);
    (((clamped - min_da) / MASS_BUCKET_WIDTH_DA) as usize).min(n - 1)
}

/// Configuration options for the [`MgfParser`].
#[derive(Debug, Clone)]
pub struct MgfParserConfig {
    /// Whether to parse and store `KEY=VALUE` metadata.
    /// Setting this to false drastically speeds up parsing and saves memory.
    pub parse_metadata: bool,
    /// Whether to use the rayon crate to parse blocks across multiple CPU cores.
    pub use_parallel_parsing: bool,
    /// How many spectra to accumulate in memory before handing off to Rayon.
    /// 10,000 is usually a great sweet spot for L3 cache and thread overhead.
    pub batch_size: usize,
}

impl Default for MgfParserConfig {
    fn default() -> Self {
        Self {
            parse_metadata: true,
            use_parallel_parsing: true,
            batch_size: 10_000,
        }
    }
}

/// One persistent shard file in a [`MgfParser`]'s low-mem shard pool: an open,
/// append-only writer plus the running byte offset new records get appended
/// at (tracked here rather than via `writer.stream_position()` since writes
/// go through a `BufWriter`).
struct ShardSlot {
    writer: BufWriter<File>,
    offset: u64,
}

/// A high-performance parser for MGF files.
///
/// Uses memory-mapping and optional parallel processing via Rayon to parse mass spectra
/// quickly and efficiently, minimizing unnecessary allocations.
pub struct MgfParser {
    /// The configuration settings used by this parser.
    pub config: MgfParserConfig,
    /// Tracks the current global spectrum ID across multiple parse calls.
    current_id: usize,
    /// Session-wide, mass-bucketed shard pool used by
    /// [`parse_file_low_mem`](Self::parse_file_low_mem), opened lazily on
    /// first use and reused (appended to) across every subsequent call for
    /// the life of this parser. Each slot is behind its own `Mutex` since,
    /// unlike a round-robin pool, which bucket a given spectrum lands in
    /// depends on its mass, not which parallel task is processing it — two
    /// tasks can and do target the same bucket concurrently.
    shard_pool: Option<Vec<Mutex<ShardSlot>>>,
}

impl Default for MgfParser {
    /// Creates a new `MgfParser` with default configuration settings.
    fn default() -> Self {
        Self {
            config: MgfParserConfig::default(),

            current_id: 0,
            shard_pool: None,
        }
    }
}
impl MgfParser {
    /// Creates a new `MgfParser` with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The parser configuration settings.
    pub fn new(config: MgfParserConfig) -> Self {
        Self {
            config,
            current_id: 0,
            shard_pool: None,
        }
    }

    /// Parses a block directly from a string slice. No intermediate allocations.
    ///
    /// Takes `parse_metadata` explicitly (rather than reading `self.config`) so
    /// this can be called from contexts that already hold a disjoint mutable
    /// borrow of another `MgfParser` field (see `parse_file_low_mem`'s shard pool).
    pub(crate) fn parse_block_slice(raw_block: &[u8], index: u64, parse_metadata: bool) -> Option<MassSpectralRecord> {
        let mut peaks = Vec::with_capacity(100);
        let mut metadata = Vec::with_capacity(10);
        let mut charge: i8 = 0;
        let mut charge_signed = false;
        let mut precursor_mz: Option<f64> = None;
        let mut is_negative = false;

        for mut line in raw_block.split(|&b| b == b'\n') {
            // Strip trailing carriage return if Windows-formatted
            if let Some(&b'\r') = line.last() {
                line = &line[..line.len() - 1];
            }
            if line.is_empty() || line.starts_with(b"BEGIN") || line.starts_with(b"END") {
                continue;
            }

            if let Some(eq_idx) = memchr::memchr(b'=', line) {
                let key = &line[..eq_idx];
                let val = &line[eq_idx + 1..];

                // Fast path for CHARGE
                if key.eq_ignore_ascii_case(b"CHARGE") {
                    let val_str = unsafe { std::str::from_utf8_unchecked(val) };
                    (charge, charge_signed) = parse_mgf_charge(val_str);
                    continue;
                }

                // Needed for charge recovery below, so read them even when
                // metadata is not kept.
                if key.eq_ignore_ascii_case(b"PEPMASS") {
                    let val_str = unsafe { std::str::from_utf8_unchecked(val) };
                    precursor_mz = val_str.split_whitespace().next().and_then(|v| v.parse::<f64>().ok());
                } else if key.eq_ignore_ascii_case(b"IONMODE")
                    || key.eq_ignore_ascii_case(b"ION_MODE")
                    || key.eq_ignore_ascii_case(b"POLARITY")
                {
                    let v = unsafe { std::str::from_utf8_unchecked(val) }.trim();
                    if v.eq_ignore_ascii_case("negative") || v.eq_ignore_ascii_case("neg") || v == "-" {
                        is_negative = true;
                    }
                }

                if parse_metadata {
                    // Only do UTF-8 unchecked conversion when we actually need a String
                    let key_str = unsafe { std::str::from_utf8_unchecked(key) };
                    let val_str = unsafe { std::str::from_utf8_unchecked(val) };
                    metadata.push((
                        CompactString::new(key_str.to_ascii_uppercase()),
                        CompactString::new(val_str),
                    ));
                }
            } else {
                // It's a peak line. Split by space or tab.
                let mut parts = line.split(|&b| b == b' ' || b == b'\t');
                if let (Some(mz_bytes), Some(int_bytes)) = (parts.next(), parts.next()) {
                    // fast-float parses &[u8] natively!
                    if let (Ok(mz), Ok(intensity)) =
                        (fast_float::parse(mz_bytes), fast_float::parse(int_bytes))
                    {
                        peaks.push(Peak::new(mz, intensity));
                    }
                }
            }
        }

        if peaks.is_empty() {
            None
        } else {
            peaks.shrink_to_fit();

            // A missing, zero or unparseable charge is only replaced when the
            // precursor isotope envelope shows it (see recover_charge); it
            // otherwise stays 0. An unsigned CHARGE (e.g. "1") takes its sign
            // from IONMODE / POLARITY.
            if charge == 0 {
                charge = recover_charge(&peaks, precursor_mz, is_negative);
            } else if !charge_signed && is_negative {
                charge = -charge;
            }
            if parse_metadata {
                metadata.push((
                    CompactString::new("CHARGE"),
                    CompactString::new(charge.to_string()),
                ));
            }

            let mut spec = MassSpectralRecord::new(peaks, charge, index as usize);
            if parse_metadata {
                metadata.shrink_to_fit();
                *spec.metadata_mut() = metadata;
            }
            Some(spec)
        }
    }

    // pub fn parse_multiple_files(
    //     &mut self,
    //     file_paths: &[&str],
    // ) -> io::Result<Vec<MassSpectralRecord>> {
    //     // PASS 1: Mmap all files and find block indices in parallel
    //     // This is strictly I/O and memory-bandwidth bound.
    //     let file_maps_and_indices: Vec<(Arc<Mmap>, Vec<usize>)> = file_paths
    //         .par_iter()
    //         .map(|&path| {
    //             let file = File::open(path).unwrap(); // Handle errors properly in production
    //             let mmap = unsafe { Arc::new(Mmap::map(&file).unwrap()) };

    //             let begin_marker = b"BEGIN IONS";
    //             let indices: Vec<usize> = memmem::find_iter(&mmap, begin_marker).collect();

    //             (mmap, indices)
    //         })
    //         .collect();

    //     // PASS 2: Calculate deterministic starting IDs (Prefix Sum)
    //     // This is sequential but takes literally microseconds.
    //     let mut tasks = Vec::new();
    //     let mut current_global_id = self.current_id as u64;

    //     for (mmap, indices) in file_maps_and_indices {
    //         let spectrum_count = indices.len() as u64;
    //         tasks.push((mmap, indices, current_global_id));
    //         current_global_id += spectrum_count;
    //     }

    //     // Update the parser's state
    //     self.current_id = current_global_id as usize;

    //     // PASS 3: Parse everything!
    //     // We flatten the results from all files into one massive, ordered vector.
    //     let all_spectra: Vec<MassSpectralRecord> = tasks
    //         .into_par_iter()
    //         .flat_map(|(mmap, indices, base_id)| {
    //             // Use the chunking strategy we discussed earlier for Rayon efficiency
    //             let chunk_size = 1000;

    //             indices.par_chunks(chunk_size).enumerate().flat_map_iter(
    //                 move |(chunk_idx, chunk_indices)| {
    //                     let mut local_spectra = Vec::with_capacity(chunk_size);
    //                     let chunk_base_id = base_id + (chunk_idx * chunk_size) as u64;

    //                     for (i, &start_idx) in chunk_indices.iter().enumerate() {
    //                         // Find the end index (start of next block or end of file)
    //                         // Note: Because we chunked the indices, we need a reliable way to get the next index.
    //                         // The easiest way is to use memmem to find the next BEGIN IONS from start_idx + 10,
    //                         // or just use the end of the mmap if it's the last one.
    //                         let end_idx = memmem::find(&mmap[start_idx + 10..], b"BEGIN IONS")
    //                             .map(|offset| start_idx + 10 + offset)
    //                             .unwrap_or(mmap.len());

    //                         let raw_block = &mmap[start_idx..end_idx];

    //                         // Call our new byte-level parser
    //                         if let Some(spec) =
    //                             self.parse_block_slice(raw_block, chunk_base_id + i as u64)
    //                         {
    //                             local_spectra.push(spec);
    //                         }
    //                     }
    //                     local_spectra
    //                 },
    //             )
    //         })
    //         .collect();

    //     Ok(all_spectra)
    // }
    /// Parses an MGF file and returns a vector of `MassSpectralRecord`s.
    ///
    /// This method uses memory-mapping to map the file into virtual memory,
    /// identifies block boundaries, and then parses them (potentially in parallel,
    /// depending on the parser configuration).
    ///
    /// # Arguments
    ///
    /// * `file_name` - The path to the MGF file.
    ///
    /// # Returns
    ///
    /// An `io::Result` containing a vector of parsed `MassSpectralRecord`s.
    pub fn parse_file(&mut self, file_name: &str) -> io::Result<Vec<MassSpectralRecord>> {
        let file = File::open(file_name)?;

        // Map the file into virtual memory
        let mmap = unsafe { Mmap::map(&file)? };

        // Estimate ~2.5KB per spectrum for initial capacity to avoid reallocations
        let estimated_spectra = mmap.len() / MGF_SPECTRA_SIZE;
        let mut start_indices = Vec::with_capacity(estimated_spectra);

        let begin_marker = b"BEGIN IONS";

        for index in memmem::find_iter(&mmap, begin_marker) {
            start_indices.push(index);
        }

        // 2. Process blocks in parallel
        let parse_metadata = self.config.parse_metadata;
        let base_id = self.current_id as u64;
        let spectra: Vec<MassSpectralRecord> = start_indices
            .par_iter()
            .enumerate()
            .filter_map(|(record_index, &start_idx)| {
                // The end of this block is just the start of the next block
                // (or the end of the file if it's the last block)
                let end_idx = start_indices
                    .get(record_index + 1)
                    .copied()
                    .unwrap_or(mmap.len());

                let raw_block = &mmap[start_idx..end_idx];
                Self::parse_block_slice(raw_block, record_index as u64 + base_id, parse_metadata)
                    .map(|mut s| {
                        s.source_byte_start = start_idx as u64;
                        s.source_byte_end = end_idx as u64;
                        s
                    })
            })
            .collect();
        self.current_id += spectra.len();

        Ok(spectra)
    }

    /// Streaming, low-memory variant of [`parse_file`](Self::parse_file).
    ///
    /// Instead of collecting every parsed spectrum into one big in-memory
    /// `Vec` (the normal path's peak RSS, dominated by that `Vec`, scales
    /// with total dataset size), each parallel worker filters and converts
    /// its own block to an [`MSNSpectrum`] and immediately serializes it
    /// (rkyv) to its own shard file under `spill_dir`, retaining only a
    /// small [`SpillKey`] per spectrum. Peak RSS during parsing is bounded
    /// by shard count and per-block working memory, not dataset size.
    ///
    /// `workflow` is applied per-spectrum here
    /// (mirroring what [`SpectralExperiment::ingest_file`] does in a batch
    /// pass after the full parse) so the set of spectra that end up spilled
    /// matches what normal ingestion would have kept.
    ///
    /// Returns `(keys for kept spectra, total blocks that parsed into a
    /// record before workflow/charge filtering)`. The second element mirrors
    /// `total_spectra_ingested`'s existing meaning from
    /// [`SpectralExperiment::ingest_file`] — "how many spectra did we
    /// attempt", not "how many survived" — matching what `parse_file` would
    /// have returned via `.len()` before any workflow filtering ran.
    ///
    /// Spectrum ids are assigned from parse order (like
    /// [`parse_file`](Self::parse_file)'s records), *not* renumbered densely
    /// over kept spectra the way `SpectralExperiment::ingest_file` renumbers
    /// `MSNSpectrum::id`. `file_index` is stamped into each spilled
    /// [`LowMemRecord`] instead, so output written from a partition later
    /// can still resolve which input file a spectrum came from.
    ///
    /// `mass_bucket_min_da`/`mass_bucket_max_da` bound the shard pool's mass
    /// range (bucket width is always [`MASS_BUCKET_WIDTH_DA`]) — pass the
    /// caller's configured mz range (e.g. `ClusterEngine::hasher`'s
    /// `min_mz`/`max_mz`) so the pool doesn't allocate shards for mass
    /// ranges the workflow's own filters would never let through anyway.
    ///
    /// [`SpectralExperiment::ingest_file`]: crate::clustering::cluster::experiment::SpectralExperiment::ingest_file
    pub fn parse_file_low_mem(
        &mut self,
        file_name: &str,
        file_index: usize,
        spill_dir: &Path,
        workflow: &WorkFlowManager,
        mass_bucket_min_da: f64,
        mass_bucket_max_da: f64,
        collect_filtered: bool,
    ) -> io::Result<(Vec<SpillKey>, usize, Vec<SpectrumMetadata>)> {
        let file = File::open(file_name)?;

        // Map the file into virtual memory
        let mmap = unsafe { Mmap::map(&file)? };

        let estimated_spectra = mmap.len() / MGF_SPECTRA_SIZE;
        let mut start_indices = Vec::with_capacity(estimated_spectra);

        let begin_marker = b"BEGIN IONS";
        for index in memmem::find_iter(&mmap, begin_marker) {
            start_indices.push(index);
        }

        std::fs::create_dir_all(spill_dir)?;

        let base_id = self.current_id as u64;
        let n = start_indices.len();
        self.current_id += n;

        if self.shard_pool.is_none() {
            self.shard_pool = Some(Self::open_shard_pool(
                spill_dir,
                mass_bucket_min_da,
                mass_bucket_max_da,
            )?);
        }
        let pool = self
            .shard_pool
            .as_ref()
            .expect("shard_pool was just initialized above if it was None");
        let parse_metadata = self.config.parse_metadata;

        // Chunking here is purely for parallel load-balancing: unlike the old
        // round-robin pool, which mass bucket a spectrum lands in has nothing
        // to do with which chunk it's parsed in, so chunk count is decoupled
        // from shard count.
        let num_chunks = (rayon::current_num_threads() * 4).max(1);
        let chunk_size = n.div_ceil(num_chunks).max(1);

        type ChunkResult = (Vec<SpillKey>, usize, Vec<SpectrumMetadata>);
        let shard_results: Vec<io::Result<ChunkResult>> = start_indices
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| -> io::Result<ChunkResult> {
                let mut shard_keys = Vec::with_capacity(chunk.len());
                let mut filtered = Vec::new();
                let mut parsed_ok = 0usize;
                let chunk_start = chunk_idx * chunk_size;

                for (local_i, &start_idx) in chunk.iter().enumerate() {
                    let global_i = chunk_start + local_i;
                    let end_idx = start_indices
                        .get(global_i + 1)
                        .copied()
                        .unwrap_or(mmap.len());
                    let raw_block = &mmap[start_idx..end_idx];
                    let record_id = base_id + global_i as u64;

                    let Some(mut record) = Self::parse_block_slice(raw_block, record_id, parse_metadata)
                    else {
                        continue;
                    };
                    // Matches parse_file/ingest_file: total_spectra_ingested
                    // counts blocks that parsed into a record (>= 1 peak
                    // line), before workflow/charge filtering runs.
                    parsed_ok += 1;

                    for op in workflow.operations() {
                        op.operate(std::slice::from_mut(&mut record), false);
                    }
                    if record.peaks().is_empty() {
                        if collect_filtered {
                            filtered.push(SpectrumMetadata::from_record(&record, global_i, file_index));
                        }
                        continue;
                    }

                    // SCANS is sometimes a range/list (e.g. MSn-to-pseudo-MS2
                    // spectra merged from several underlying scans, "185-238,
                    // 737-739,..."), which doesn't parse as a plain integer;
                    // the output identifier then falls back to `index`.
                    let scan_number = record
                        .get_metadata("SCANS")
                        .and_then(|s| s.parse::<usize>().ok());
                    let title = record.get_metadata("TITLE").unwrap_or("").to_string();
                    let msn = record.to_msn();
                    let neutral_mass = msn.get_neutral_mass();
                    let low_mem_record = LowMemRecord {
                        msn,
                        scan_number,
                        index: global_i,
                        title,
                        file_index,
                    };
                    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&low_mem_record)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

                    let shard_idx =
                        mass_bucket_index(neutral_mass, mass_bucket_min_da, mass_bucket_max_da);
                    let offset = {
                        let mut slot = pool[shard_idx]
                            .lock()
                            .expect("shard slot mutex poisoned by a panicking writer thread");
                        slot.writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
                        slot.writer.write_all(&bytes)?;
                        let offset = slot.offset;
                        slot.offset += 4 + bytes.len() as u64;
                        offset
                    };

                    shard_keys.push(SpillKey {
                        id: low_mem_record.msn.id,
                        charge: low_mem_record.msn.charge(),
                        precursor_mz: low_mem_record.msn.precursor_mz,
                        neutral_mass,
                        shard: shard_idx as u32,
                        offset,
                        len: bytes.len() as u32,
                    });
                }
                Ok((shard_keys, parsed_ok, filtered))
            })
            .collect();

        // Every bucket's BufWriter must be flushed before finalize_low_mem_partitions
        // reopens shard files for reading — flush is cheap (userspace buffer
        // -> OS page cache only), so doing it for the whole pool after every
        // input file is negligible next to the actual parse/write work.
        for slot in pool.iter() {
            slot.lock()
                .expect("shard slot mutex poisoned by a panicking writer thread")
                .writer
                .flush()?;
        }

        let mut keys = Vec::with_capacity(n);
        let mut total_parsed = 0usize;
        let mut filtered = Vec::new();
        for shard_result in shard_results {
            let (shard_keys, parsed_ok, shard_filtered) = shard_result?;
            keys.extend(shard_keys);
            total_parsed += parsed_ok;
            filtered.extend(shard_filtered);
        }
        Ok((keys, total_parsed, filtered))
    }

    /// Opens this parser's low-mem shard pool: a small, fixed number of
    /// `shard_{idx:06}.bin` files under `spill_dir`, created once and then
    /// reused (appended to) by every subsequent [`parse_file_low_mem`](Self::parse_file_low_mem)
    /// call for the life of this parser.
    ///
    /// Shard count first scaled with `rayon::current_num_threads() * 4`
    /// *per file*, with a globally-incrementing id so no two files ever
    /// shared a shard — total shard files scaled with the number of input
    /// files (a 3,721-file corpus produced ~315,000 shard files, almost all
    /// tiny). A fixed session-wide pool with round-robin-by-parse-order
    /// assignment fixed that (shard count constant regardless of corpus
    /// size), but left every shard's contents an unbiased sample of the
    /// whole mass range — so `finalize_low_mem_partitions`, which reads
    /// records back out sorted by mass, still had to touch nearly every
    /// shard for nearly every partition once shard count dropped from
    /// hundreds of thousands to a few dozen. On an HDD (seek-dominated
    /// random reads) this stalled at corpus scale even though it looked fine
    /// on a page-cache-sized subset.
    ///
    /// This pool instead assigns each spectrum to a shard by its
    /// [`mass_bucket_index`] (a fixed-width, 1 Da mass bucket — see
    /// [`MASS_BUCKET_WIDTH_DA`]), so a clustering partition's narrow mass
    /// window almost always maps to a single shard instead of all of them.
    /// `min_da`/`max_da` bound the range (see `parse_file_low_mem`'s docs) —
    /// independent of both corpus size and thread count, and small as long as
    /// the caller's mz range is reasonably tight, which keeps total shard
    /// count bounded while still giving each partition read locality.
    /// Because bucket assignment depends on a spectrum's mass rather than
    /// which parallel chunk parsed it, two threads can target the same shard
    /// concurrently, so each slot is individually `Mutex`-guarded rather than
    /// exclusively owned per task.
    fn open_shard_pool(spill_dir: &Path, min_da: f64, max_da: f64) -> io::Result<Vec<Mutex<ShardSlot>>> {
        let num_shards = num_mass_buckets(min_da, max_da);
        (0..num_shards)
            .map(|shard_idx| -> io::Result<Mutex<ShardSlot>> {
                let shard_path = spill_dir.join(format!("shard_{shard_idx:06}.bin"));
                Ok(Mutex::new(ShardSlot {
                    writer: BufWriter::new(File::create(&shard_path)?),
                    offset: 0,
                }))
            })
            .collect()
    }
}

/// A lightweight per-spectrum key produced by [`MgfParser::parse_file_low_mem`]:
/// enough to decide m/z/charge partition membership (via
/// [`crate::clustering::cluster::mz_partition::mz_charge_boundaries`])
/// without needing the spectrum's peak data resident in memory. The full
/// record lives in `spill_dir/shard_{shard:06}.bin`, as a `u32` little-endian
/// length prefix followed by that many bytes of rkyv-serialized
/// [`LowMemRecord`], starting at byte `offset` in that file. `shard` indexes
/// a small, fixed, session-wide, mass-bucketed pool (see
/// `MgfParser::open_shard_pool`) that every ingested file appends its
/// records into — unlike a per-file shard id, the same `shard` value is
/// shared across many different input files (and, since assignment is by
/// mass rather than by file, records from the same file usually end up
/// scattered across several shards), which is fine since `offset` still
/// pins each record to its own bytes.
#[derive(Clone, Copy, Debug)]
pub struct SpillKey {
    pub id: u64,
    pub charge: i8,
    pub precursor_mz: f64,
    pub neutral_mass: f64,
    pub shard: u32,
    pub offset: u64,
    pub len: u32,
}

/// What actually gets serialized to a shard/partition file: an `MSNSpectrum`
/// plus the two extra fields normal ingestion tracks in
/// `SpectralExperiment::metadata` (`scan_number`, `file_index`) but which
/// have no dense-id-indexed table to live in on the low-mem path — so they
/// travel with the record itself instead.
#[derive(Clone, Debug, Archive, Serialize, Deserialize)]
pub struct LowMemRecord {
    pub msn: MSNSpectrum,
    pub scan_number: Option<usize>,
    /// 0-based position of the spectrum in its source file.
    pub index: usize,
    pub title: String,
    pub file_index: usize,
}

/// A high-performance writer for MGF files.
///
/// Supports both sequential and parallel writing of mass spectra,
/// serializing them into standard MGF plain-text blocks.
pub struct MgfWriter;

impl Default for MgfWriter {
    /// Creates a new `MgfWriter` using default settings.
    fn default() -> Self {
        Self::new()
    }
}

impl MgfWriter {
    /// Creates a new `MgfWriter`.
    pub fn new() -> Self {
        Self
    }

    /// Writes a list of MassSpectralRecords to any `io::Write` buffer sequentially.
    pub fn write<W: Write>(&self, writer: W, spectra: &[MassSpectralRecord]) -> io::Result<()> {
        let mut buf_writer = BufWriter::new(writer);

        for spectrum in spectra {
            self.write_single_spectrum(&mut buf_writer, spectrum)?;
        }

        buf_writer.flush()?;
        Ok(())
    }

    /// Writes a list of spectra to a file sequentially.
    pub fn write_to_file(&self, file_name: &str, spectra: &[MassSpectralRecord]) -> io::Result<()> {
        let file = File::create(file_name)?;
        self.write(file, spectra)
    }

    /// High-performance writer that formats MGF blocks in parallel using Rayon.
    /// It builds string chunks concurrently and writes them sequentially to prevent I/O bottlenecks.
    pub fn write_parallel<W: Write>(
        &self,
        writer: W,
        spectra: &[MassSpectralRecord],
        chunk_size: usize,
    ) -> io::Result<()> {
        let mut buf_writer = BufWriter::new(writer);

        // 1. Format spectra into String blocks in parallel
        let formatted_chunks: Vec<String> = spectra
            .par_chunks(chunk_size)
            .map(|chunk| {
                // Pre-allocate assuming ~100 bytes per spectrum to minimize reallocation
                let mut chunk_string = String::with_capacity(chunk.len() * 1024);

                for spectrum in chunk {
                    chunk_string.push_str("BEGIN IONS\n");

                    // Write metadata keys (TITLE, PEPMASS, CHARGE, etc.)
                    for (key, value) in spectrum.metadata() {
                        chunk_string.push_str(key);
                        chunk_string.push('=');
                        chunk_string.push_str(value);
                        chunk_string.push('\n');
                    }
                    // Inside your loop:
                    for peak in spectrum.peaks() {
                        let _ = write!(chunk_string, "{} {}\n", peak.mz, peak.intensity);
                    }

                    chunk_string.push_str("END IONS\n");
                }
                chunk_string
            })
            .collect();

        // 2. Write the pre-formatted strings to the IO buffer sequentially
        for chunk_str in formatted_chunks {
            buf_writer.write_all(chunk_str.as_bytes())?;
        }

        buf_writer.flush()?;
        Ok(())
    }

    /// Writes a list of spectra to a file in parallel.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The destination path for the MGF file.
    /// * `spectra` - A slice of `MassSpectralRecord`s to write.
    pub fn write_parallel_to_file(
        &self,
        file_name: &str,
        spectra: &[MassSpectralRecord],
    ) -> io::Result<()> {
        let file = File::create(file_name)?;
        // 10,000 matches your parser's sweet spot for L3 cache and thread overhead
        self.write_parallel(file, spectra, 10_000)
    }

    /// Helper to write a single spectrum block
    #[inline]
    fn write_single_spectrum<W: Write>(
        &self,
        writer: &mut W,
        spectrum: &MassSpectralRecord,
    ) -> io::Result<()> {
        writer.write_all(b"BEGIN IONS\n")?;

        for (key, value) in spectrum.metadata() {
            writeln!(writer, "{}={}", key, value)?;
        }

        for peak in spectrum.peaks() {
            writeln!(writer, "{} {}", peak.mz, peak.intensity)?;
        }

        writer.write_all(b"END IONS\n")?;
        Ok(())
    }
}
#[cfg(test)]
mod mgf_parser_tests {
    use super::*;
    use std::io::Cursor;

    const MGF_DATA: &str = "\
        BEGIN IONS\n\
        TITLE=Spectrum 1\n\
        PEPMASS=500.25\n\
        100.0 1000.0\n\
        200.0 500.0\n\
        END IONS\n\
        BEGIN IONS\n\
        PEPMASS=500.25\n\
        TITLE=Spectrum 2\n\
        300.0 800.0\n\
        END IONS\n\
    ";
    const FILE_NAME: &str = "test_files/test2.mgf";

    fn block_charge(block: &str, parse_metadata: bool) -> i8 {
        MgfParser::parse_block_slice(block.as_bytes(), 0, parse_metadata)
            .unwrap()
            .get_charge()
    }

    #[test]
    fn charge_formats_parse() {
        for (raw, expected) in [
            ("2", (2, false)),
            ("2+", (2, true)),
            ("+2", (2, true)),
            ("1-", (-1, true)),
            ("-1", (-1, true)),
            ("2.0", (2, false)),
            ("2.0+", (2, true)),
            ("1.0-", (-1, true)),
            (" 3+ ", (3, true)),
            ("2+ and 3+", (2, true)),
            ("2+,3+", (2, true)),
            ("0", (0, false)),
            ("0+", (0, false)),
            ("1.5", (0, false)),
            ("", (0, false)),
            ("NaN", (0, false)),
            ("abc", (0, false)),
            ("300", (0, false)),
        ] {
            assert_eq!(parse_mgf_charge(raw), expected, "CHARGE={raw:?}");
        }
    }

    #[test]
    fn zero_or_missing_charge_stays_unknown_without_isotope_evidence() {
        let neg_zero = "BEGIN IONS\nPEPMASS=300.0\nIONMODE=Negative\nCHARGE=0\n100.0 10.0\nEND IONS\n";
        let neg_missing = "BEGIN IONS\nPEPMASS=300.0\nIONMODE=negative\n100.0 10.0\nEND IONS\n";
        let pos_zero = "BEGIN IONS\nPEPMASS=300.0\nIONMODE=Positive\nCHARGE=0\n100.0 10.0\nEND IONS\n";
        let neg_z2 = "BEGIN IONS\nPEPMASS=500.0\nIONMODE=negative\nCHARGE=0\n100.0 10.0\n500.5017 25.0\nEND IONS\n";
        for parse_metadata in [true, false] {
            assert_eq!(block_charge(neg_zero, parse_metadata), 0);
            assert_eq!(block_charge(neg_missing, parse_metadata), 0);
            assert_eq!(block_charge(pos_zero, parse_metadata), 0);
            assert_eq!(block_charge(neg_z2, parse_metadata), -2);
        }
    }

    #[test]
    fn unsigned_charge_takes_sign_from_ionmode_but_explicit_sign_wins() {
        let unsigned = "BEGIN IONS\nPEPMASS=300.0\nCHARGE=1.0\nIONMODE=negative\n100.0 10.0\nEND IONS\n";
        let explicit = "BEGIN IONS\nPEPMASS=300.0\nCHARGE=1+\nIONMODE=negative\n100.0 10.0\nEND IONS\n";
        assert_eq!(block_charge(unsigned, true), -1);
        assert_eq!(block_charge(explicit, true), 1);
    }

    #[test]
    fn zero_charge_is_recovered_from_isotope_envelope_without_metadata() {
        let z2 = "BEGIN IONS\nPEPMASS=500.0\nCHARGE=0\n100.0 10.0\n500.0 50.0\n500.5017 25.0\nEND IONS\n";
        assert_eq!(block_charge(z2, true), 2);
        assert_eq!(block_charge(z2, false), 2);
    }

    #[test]
    fn recovered_charge_is_written_to_metadata_once() {
        let block = "BEGIN IONS\nPEPMASS=500.0\nIONMODE=negative\nCHARGE=0\n100.0 10.0\n500.5017 25.0\nEND IONS\n";
        let rec = MgfParser::parse_block_slice(block.as_bytes(), 0, true).unwrap();
        let charges: Vec<_> = rec.metadata().iter().filter(|(k, _)| k == "CHARGE").collect();
        assert_eq!(charges.len(), 1);
        assert_eq!(charges[0].1, "-2");
    }

    #[test]
    #[ignore = "needs test_files/test2.mgf (600,566 spectra, not in git); see README \"Tests\""]
    fn test_sequential_with_metadata() {
        let mut config = MgfParserConfig::default();
        config.use_parallel_parsing = false;
        config.parse_metadata = true;

        let mut parser = MgfParser::new(config);
        let spectra = parser.parse_file(FILE_NAME).unwrap();

        assert_eq!(spectra.len(), 600566);
        assert_eq!(spectra[0].get_metadata("TITLE").unwrap(), "Spectrum 1");
    }

    #[test]
    #[ignore = "needs test_files/test2.mgf (600,566 spectra, not in git); see README \"Tests\""]
    fn test_parallel_without_metadata() {
        let mut config = MgfParserConfig::default();
        config.use_parallel_parsing = true;
        config.parse_metadata = false; // Turn metadata off

        let mut parser = MgfParser::new(config);
        let spectra = parser.parse_file(FILE_NAME).unwrap();

        assert_eq!(spectra.len(), 600566);
        // Metadata map should be completely empty to save allocations
        assert!(spectra[0].metadata().is_empty());
        // But peaks should still parse correctly
        assert_eq!(spectra[0].peaks().len(), 189);
    }
}
