//! High-performance mzML parsing.
//!
//! mzML is a harder parsing problem than MGF: spectra are XML elements, peak
//! data is base64-encoded (often zlib-compressed) binary float arrays, and
//! per-spectrum metadata lives in verbose CV-parameter elements rather than
//! flat `KEY=VALUE` lines. To stay in the same performance tier as
//! [`crate::io::mgf::MgfParser`], this module mirrors its architecture
//! exactly rather than reaching for a general-purpose XML/mzML library:
//!
//! - The file is memory-mapped, never fully read into an owned buffer.
//! - Spectrum boundaries are found up front, then every spectrum is parsed
//!   independently and in parallel (via rayon) -- the same "find offsets,
//!   then parse each block on its own thread" shape as the MGF parser's
//!   `BEGIN IONS` scan.
//! - Each spectrum's XML fragment is walked with a pull/SAX parser
//!   ([`quick_xml`]), never materialized into a DOM tree.
//! - Binary peak data decodes straight from base64 + (optional) zlib bytes
//!   into `f32`/`f64` via `from_le_bytes` -- no per-character ASCII float
//!   parsing the way MGF's peak lines need, since mzML's numbers are already
//!   binary.
//!
//! Two ways to find spectrum boundaries are supported:
//!
//! 1. **Indexed fast path**: real-world mzML is usually `indexedmzML`, which
//!    carries a trailing `<indexList>` of exact byte offsets to every
//!    `<spectrum>` element (part of the format spec, written by essentially
//!    every real acquisition/conversion tool). When present, this is read
//!    directly -- no need to scan the file body for boundaries at all,
//!    exactly analogous to using MGF's `BEGIN IONS` scan except we don't
//!    even need to do the scan.
//! 2. **Fallback scan path**: for plain `<mzML>` (no index) or a corrupt/
//!    unparseable index, `memchr` scans for `<spectrum ` occurrences the
//!    same way the MGF parser scans for `BEGIN IONS`, and each spectrum's
//!    end is simply the next one's start (or EOF for the last).
//!
//! Both paths produce the same `Vec<(start, end)>` byte-range list, which
//! feeds the same parallel per-spectrum parse either way.
use crate::spectrum::{MassSpectralRecord, Peak};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use memchr::memmem;
use memmap2::Mmap;
use quick_xml::Reader;
use quick_xml::events::Event;
use rayon::prelude::*;
use std::fs::File;
use std::io;


/// Configuration options for [`MzmlParser`]. Mirrors
/// [`crate::io::mgf::MgfParserConfig`]'s knobs.
#[derive(Debug, Clone)]
pub struct MzmlParserConfig {
    /// Whether to parse and store spectrum-level metadata (scan id, extra
    /// cvParams beyond what's needed for precursor m/z/charge/RT). Disabling
    /// this skips a nontrivial amount of string work per spectrum.
    pub parse_metadata: bool,
    /// Whether to parse spectra across multiple CPU cores via rayon.
    pub use_parallel_parsing: bool,
    /// Use the hand-rolled mzML-specific tokenizer (`parse_spectrum_slice_fast`)
    /// instead of the quick_xml-based one (`parse_spectrum_slice_quickxml`).
    /// Profiling found the general XML event loop was ~43% of per-spectrum
    /// time even after fixing allocation waste and switching decompressors
    /// elsewhere -- the hand-rolled tokenizer exists specifically to cut
    /// that down, by skipping everything a general XML parser has to handle
    /// for arbitrary documents that a `<spectrum>` fragment never needs
    /// (entity unescaping, full quote-style/namespace generality). Defaults
    /// to `true`; set `false` to fall back to the more general, more
    /// battle-tested quick_xml path if the fast one ever mis-parses some
    /// real-world mzML variant it wasn't tested against -- both paths are
    /// cross-checked for identical output on every real file this was
    /// developed against (see the `tests` module).
    pub use_fast_tokenizer: bool,
}

impl Default for MzmlParserConfig {
    fn default() -> Self {
        Self {
            parse_metadata: true,
            use_parallel_parsing: true,
            use_fast_tokenizer: true,
        }
    }
}

/// A high-performance parser for mzML files. See the module docs for the
/// architecture; usage mirrors [`crate::io::mgf::MgfParser`].
pub struct MzmlParser {
    pub config: MzmlParserConfig,
    current_id: usize,
}

impl Default for MzmlParser {
    fn default() -> Self {
        Self::new(MzmlParserConfig::default())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArrayKind {
    Mz,
    Intensity,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
    F32,
    F64,
}

impl MzmlParser {
    pub fn new(config: MzmlParserConfig) -> Self {
        Self {
            config,
            current_id: 0,
        }
    }

    /// Parses an mzML file and returns a vector of `MassSpectralRecord`s.
    ///
    /// Memory-maps the file, finds every spectrum's byte range (indexed fast
    /// path if the file is `indexedmzML` with a valid trailing index,
    /// otherwise a `memchr` scan for `<spectrum `), then parses each range
    /// independently, in parallel if configured.
    pub fn parse_file(&mut self, file_name: &str) -> io::Result<Vec<MassSpectralRecord>> {
        let file = File::open(file_name)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let spans = Self::spectrum_spans(&mmap);
        let base_id = self.current_id;
        let parse_metadata = self.config.parse_metadata;
        let parse_one: fn(&[u8], usize, bool) -> Option<MassSpectralRecord> =
            if self.config.use_fast_tokenizer {
                parse_spectrum_slice_fast
            } else {
                parse_spectrum_slice_quickxml
            };

        let records: Vec<MassSpectralRecord> = if self.config.use_parallel_parsing {
            spans
                .par_iter()
                .enumerate()
                .filter_map(|(i, &(start, end))| {
                    parse_one(&mmap[start..end], base_id + i, parse_metadata).map(|mut r| {
                        r.source_byte_start = start as u64;
                        r.source_byte_end = end as u64;
                        r
                    })
                })
                .collect()
        } else {
            spans
                .iter()
                .enumerate()
                .filter_map(|(i, &(start, end))| {
                    parse_one(&mmap[start..end], base_id + i, parse_metadata).map(|mut r| {
                        r.source_byte_start = start as u64;
                        r.source_byte_end = end as u64;
                        r
                    })
                })
                .collect()
        };

        self.current_id += records.len();
        Ok(records)
    }

    /// Finds every spectrum's `[start, end)` byte range in `mmap`: the
    /// indexed fast path if available and parseable, the `memchr` scan
    /// otherwise. Never fails outright -- a broken index just falls back to
    /// scanning, same as a plain (non-indexed) file would.
    pub fn spectrum_spans(mmap: &[u8]) -> Vec<(usize, usize)> {
        if let Some(spans) = Self::indexed_spans(mmap) {
            return spans;
        }
        Self::scanned_spans(mmap)
    }

    /// Reads the trailing `<indexList>` an `indexedmzML` file carries,
    /// returning byte offsets for every `<spectrum>` element it lists. Each
    /// span's end is simply the next span's start (or EOF for the last) --
    /// safe even if that includes trailing bytes past `</spectrum>`, since
    /// the per-spectrum XML parse below stops at the matching end tag and
    /// ignores whatever follows in its slice.
    fn indexed_spans(mmap: &[u8]) -> Option<Vec<(usize, usize)>> {
        let marker = b"<indexListOffset>";
        let marker_pos = memmem::rfind(mmap, marker)?;
        let value_start = marker_pos + marker.len();
        let value_end = memmem::find(&mmap[value_start..], b"</indexListOffset>")? + value_start;
        let offset_str = std::str::from_utf8(&mmap[value_start..value_end]).ok()?;
        let index_list_offset: usize = offset_str.trim().parse().ok()?;
        if index_list_offset >= mmap.len() {
            return None;
        }

        let mut reader = Reader::from_reader(&mmap[index_list_offset..]);
        reader.config_mut().trim_text(true);

        let mut offsets = Vec::new();
        let mut in_spectrum_index = false;
        let mut in_offset_tag = false;
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let name = e.local_name();
                    if name.as_ref() == b"index" {
                        in_spectrum_index = e.attributes().flatten().any(|a| {
                            a.key.as_ref() == b"name" && a.value.as_ref() == b"spectrum"
                        });
                    } else if in_spectrum_index && name.as_ref() == b"offset" {
                        in_offset_tag = true;
                    }
                }
                Ok(Event::Text(t)) if in_offset_tag => {
                    if let Ok(text) = t.unescape() {
                        if let Ok(v) = text.trim().parse::<usize>() {
                            offsets.push(v);
                        }
                    }
                }
                Ok(Event::End(e)) => {
                    let name = e.local_name();
                    if name.as_ref() == b"offset" {
                        in_offset_tag = false;
                    } else if name.as_ref() == b"index" {
                        in_spectrum_index = false;
                    } else if name.as_ref() == b"indexList" {
                        break;
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => return None,
                _ => {}
            }
            buf.clear();
        }

        if offsets.is_empty() {
            return None;
        }
        // Index entries are written in document order already, but sort
        // defensively -- an out-of-order index would otherwise silently
        // corrupt every span after the first swap.
        offsets.sort_unstable();
        offsets.dedup();

        let mut spans = Vec::with_capacity(offsets.len());
        for i in 0..offsets.len() {
            let start = offsets[i];
            if start >= mmap.len() {
                continue;
            }
            let end = offsets.get(i + 1).copied().unwrap_or(mmap.len());
            spans.push((start, end));
        }
        Some(spans)
    }

    /// Fallback for non-indexed mzML (or an unparseable index): scans for
    /// every `<spectrum ` occurrence the same way the MGF parser scans for
    /// `BEGIN IONS`.
    fn scanned_spans(mmap: &[u8]) -> Vec<(usize, usize)> {
        let starts: Vec<usize> = memmem::find_iter(mmap, b"<spectrum ").collect();
        starts
            .iter()
            .enumerate()
            .map(|(i, &start)| {
                let end = starts.get(i + 1).copied().unwrap_or(mmap.len());
                (start, end)
            })
            .collect()
    }
}

/// Parses one `<spectrum ...> ... </spectrum>` XML fragment into a
/// `MassSpectralRecord` using `quick_xml`'s general pull parser. Returns
/// `None` for spectra with no peaks (mirrors the MGF parser dropping empty
/// blocks). Kept as the fallback path -- see `MzmlParserConfig::use_fast_tokenizer`
/// and `parse_spectrum_slice_fast`, which is the default.
fn parse_spectrum_slice_quickxml(
    slice: &[u8],
    id: usize,
    parse_metadata: bool,
) -> Option<MassSpectralRecord> {
    let mut reader = Reader::from_reader(slice);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();

    let mut scan_number: Option<usize> = None;
    let mut precursor_mz: Option<f64> = None;
    let mut charge: i8 = 0;
    let mut polarity: i8 = 0;
    let mut isolation_target: Option<f64> = None;
    let mut isolation_lower: Option<f64> = None;
    let mut isolation_upper: Option<f64> = None;
    let mut collision_energy: Option<f64> = None;
    let mut retention_time_seconds: Option<f64> = None;
    let mut ms_level: u8 = 0;
    // mzML's own peak count for this spectrum, when present -- lets
    // decode_binary preallocate its decompression buffer at the exact
    // final size instead of guessing, see decode_binary's docs.
    let mut default_array_length: Option<usize> = None;

    let mut in_precursor = false;
    let mut current_array_kind = ArrayKind::Other;
    let mut current_precision = Precision::F64;
    let mut current_compressed = false;
    let mut current_numpress = Numpress::None;
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();
    let mut in_binary = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = e.local_name();
                match name.as_ref() {
                    b"spectrum" if scan_number.is_none() => {
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"id" => {
                                    if let Ok(id_str) = std::str::from_utf8(&attr.value) {
                                        scan_number = extract_scan_number(id_str);
                                    }
                                }
                                b"defaultArrayLength" => {
                                    default_array_length = std::str::from_utf8(&attr.value)
                                        .ok()
                                        .and_then(|s| s.parse().ok());
                                }
                                _ => {}
                            }
                        }
                    }
                    b"precursor" => in_precursor = true,
                    b"binaryDataArray" => {
                        current_array_kind = ArrayKind::Other;
                        current_precision = Precision::F64;
                        current_compressed = false;
                        current_numpress = Numpress::None;
                    }
                    b"binary" => in_binary = true,
                    b"cvParam" => {
                        // Zero heap allocation: accession is copied into a
                        // fixed-size stack buffer (accession codes are always
                        // short, e.g. "MS:1000744"), and `value`/
                        // `unitAccession` are only looked up (via
                        // `attr_value_f64`/`attr_value_eq` below, which parse
                        // straight from quick_xml's own borrowed bytes) for
                        // the handful of accessions that actually need them.
                        // Previously this allocated three owned Strings per
                        // cvParam regardless of whether any of them were even
                        // looked at -- real, measurable waste across the
                        // ~8-10 cvParams in every one of potentially millions
                        // of spectra.
                        let mut accession_buf = [0u8; 24];
                        let mut accession_len = 0usize;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"accession" {
                                let v = attr.value.as_ref();
                                let n = v.len().min(accession_buf.len());
                                accession_buf[..n].copy_from_slice(&v[..n]);
                                accession_len = n;
                                break;
                            }
                        }
                        match &accession_buf[..accession_len] {
                            b"MS:1000130" => polarity = 1,
                            b"MS:1000129" => polarity = -1,
                            b"MS:1000827" if in_precursor => isolation_target = attr_value_f64(&e, b"value"),
                            b"MS:1000828" if in_precursor => isolation_lower = attr_value_f64(&e, b"value"),
                            b"MS:1000829" if in_precursor => isolation_upper = attr_value_f64(&e, b"value"),
                            b"MS:1000045" if in_precursor => collision_energy = attr_value_f64(&e, b"value"),
                            // selected ion m/z
                            b"MS:1000744" if in_precursor => {
                                precursor_mz = attr_value_f64(&e, b"value");
                            }
                            // charge state
                            b"MS:1000041" if in_precursor => {
                                charge = attr_value_f64(&e, b"value")
                                    .map(|v| v as i8)
                                    .unwrap_or(0);
                            }
                            // scan start time
                            b"MS:1000016" => {
                                if let Some(t) = attr_value_f64(&e, b"value") {
                                    let is_minutes =
                                        attr_value_eq(&e, b"unitAccession", b"UO:0000031");
                                    retention_time_seconds = Some(if is_minutes {
                                        t * 60.0 // minutes -> seconds
                                    } else {
                                        t // UO:0000010 (second), or unspecified: assume seconds
                                    });
                                }
                            }
                            // ms level
                            b"MS:1000511" if !in_precursor => {
                                if let Some(l) = attr_value_f64(&e, b"value") {
                                    ms_level = l as u8;
                                }
                            }
                            // "MS1 spectrum" boolean cvParam -- fallback
                            b"MS:1000579" if ms_level == 0 => ms_level = 1,
                            // 64-bit float
                            b"MS:1000523" => current_precision = Precision::F64,
                            // 32-bit float
                            b"MS:1000521" => current_precision = Precision::F32,
                            // MS-Numpress positive-integer, with and without zlib.
                            // Anything decoded as plain floats here would come out
                            // the wrong length and silently drop the spectrum, so
                            // these are recognised explicitly.
                            b"MS:1002745" => current_numpress = Numpress::Pic,
                            b"MS:1002747" => {
                                current_numpress = Numpress::Pic;
                                current_compressed = true;
                            }
                            b"MS:1002312" => current_numpress = Numpress::Linear,
                            b"MS:1002746" => {
                                current_numpress = Numpress::Linear;
                                current_compressed = true;
                            }
                            b"MS:1002314" => current_numpress = Numpress::Slof,
                            b"MS:1002748" => {
                                current_numpress = Numpress::Slof;
                                current_compressed = true;
                            }
                            // zlib compression
                            b"MS:1000574" => current_compressed = true,
                            // no compression
                            b"MS:1000576" => current_compressed = false,
                            // m/z array
                            b"MS:1000514" => current_array_kind = ArrayKind::Mz,
                            // intensity array
                            b"MS:1000515" => current_array_kind = ArrayKind::Intensity,
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) if in_binary && current_array_kind != ArrayKind::Other => {
                if let Ok(unescaped) = t.unescape() {
                    match current_array_kind {
                        ArrayKind::Mz => {
                            if current_numpress != Numpress::None {
                                if let Ok(raw) = decode_binary_bytes(
                                    unescaped.as_bytes(),
                                    current_compressed,
                                    Precision::F64,
                                    None,
                                ) {
                                    mz_array = numpress_decode(current_numpress, &raw);
                                }
                            } else if let Ok(v) = decode_binary_f64(
                                unescaped.as_bytes(),
                                current_compressed,
                                current_precision,
                                default_array_length,
                            ) {
                                mz_array = v;
                            }
                        }
                        ArrayKind::Intensity => {
                            if current_numpress != Numpress::None {
                                if let Ok(raw) = decode_binary_bytes(
                                    unescaped.as_bytes(),
                                    current_compressed,
                                    Precision::F32,
                                    None,
                                ) {
                                    intensity_array = numpress_decode(current_numpress, &raw)
                                        .into_iter()
                                        .map(|v| v as f32)
                                        .collect();
                                }
                            } else if let Ok(v) = decode_binary_f32(
                                unescaped.as_bytes(),
                                current_compressed,
                                current_precision,
                                default_array_length,
                            ) {
                                intensity_array = v;
                            }
                        }
                        ArrayKind::Other => unreachable!(),
                    }
                }
            }
            Ok(Event::End(e)) => match e.local_name().as_ref() {
                b"precursor" => in_precursor = false,
                b"binary" => in_binary = false,
                b"spectrum" => break,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => return None,
            _ => {}
        }
        buf.clear();
    }

    let mut record = build_record(
        id,
        if polarity < 0 { -charge.abs() } else { charge },
        polarity,
        mz_array,
        intensity_array,
        parse_metadata,
        precursor_mz,
        retention_time_seconds,
        scan_number,
        if ms_level == 0 { 2 } else { ms_level },
    )?;
    if parse_metadata {
        if polarity != 0 { record.insert_metadata("IONMODE", if polarity < 0 { "Negative" } else { "Positive" }); }
        for (key, value) in [("ISOLATION_TARGET", isolation_target), ("ISOLATION_LOWER", isolation_lower),
                             ("ISOLATION_UPPER", isolation_upper), ("COLLISION_ENERGY", collision_energy)] {
            if let Some(v) = value { record.insert_metadata(key, &v.to_string()); }
        }
    }
    Some(record)
}

/// Finds `<cvParam ... key="...">`'s value attribute and parses it as `f64`,
/// entirely from quick_xml's own borrowed bytes -- no `String` allocated for
/// attributes that don't match, and the one that does never outlives this
/// call (it's parsed to an owned `f64` before the borrow ends).
fn attr_value_f64(e: &quick_xml::events::BytesStart, key: &[u8]) -> Option<f64> {
    e.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == key)
        .and_then(|a| std::str::from_utf8(a.value.as_ref()).ok()?.parse().ok())
}

/// Finds `<cvParam ... key="...">`'s value attribute and compares it
/// byte-for-byte against `expected`, with the same zero-allocation shape as
/// [`attr_value_f64`].
fn attr_value_eq(e: &quick_xml::events::BytesStart, key: &[u8], expected: &[u8]) -> bool {
    e.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == key)
        .is_some_and(|a| a.value.as_ref() == expected)
}

/// Builds the final `MassSpectralRecord` from what either tokenizer
/// collected -- shared so the two implementations can't silently diverge on
/// this part. Returns `None` for spectra with no peaks (mirrors the MGF
/// parser dropping empty blocks).
fn build_record(
    id: usize,
    charge: i8,
    polarity: i8,
    mz_array: Vec<f64>,
    intensity_array: Vec<f32>,
    parse_metadata: bool,
    precursor_mz: Option<f64>,
    retention_time_seconds: Option<f64>,
    scan_number: Option<usize>,
    ms_level: u8,
) -> Option<MassSpectralRecord> {
    if mz_array.is_empty() || mz_array.len() != intensity_array.len() {
        return None;
    }

    let peaks: Vec<Peak> = mz_array
        .into_iter()
        .zip(intensity_array)
        .map(|(mz, intensity)| Peak::new(mz, intensity))
        .collect();
    // Many metabolomics mzML files carry no charge cvParam; recover it like
    // the MGF parser does, but only when the polarity is actually known.
    let charge = if charge == 0 && polarity != 0 && ms_level >= 2 {
        crate::io::mgf::recover_charge(&peaks, precursor_mz, polarity < 0)
    } else {
        charge
    };
    let mut record = MassSpectralRecord::new(peaks, charge, id);
    record.ms_level = ms_level;

    if parse_metadata {
        if let Some(mz) = precursor_mz {
            record.insert_metadata("PEPMASS", &mz.to_string());
        }
        if let Some(rt) = retention_time_seconds {
            record.insert_metadata("RTINSECONDS", &rt.to_string());
        }
        if let Some(scan) = scan_number {
            record.insert_metadata("SCANS", &scan.to_string());
        }
    } else {
        // PEPMASS/RTINSECONDS/charge are load-bearing for downstream code
        // (`to_msn`, clustering partitioning, MS1 survey-scan lookup) even when
        // `parse_metadata` is off -- only genuinely optional extra metadata is
        // skipped. RT is written whether or not there is a precursor, so an MS1
        // survey scan still carries its retention time.
        if let Some(mz) = precursor_mz {
            record.insert_metadata("PEPMASS", &mz.to_string());
        }
        if let Some(rt) = retention_time_seconds {
            record.insert_metadata("RTINSECONDS", &rt.to_string());
        }
    }

    Some(record)
}

/// Hand-rolled, mzML-`<spectrum>`-specific tokenizer: a minimal byte-scanner
/// that understands only what a spectrum fragment actually contains (start/
/// end/self-closing tags, their attributes, and `<binary>` text), instead of
/// a general-purpose XML parser's full feature set. Profiling found the
/// quick_xml event loop was ~43% of per-spectrum time -- comparable to zlib
/// inflate itself -- even after removing allocation waste and switching
/// decompressors; this targets that phase specifically by skipping what a
/// spectrum fragment never needs:
/// - **No entity unescaping anywhere.** Accession codes, numeric attribute
///   values, and base64 `<binary>` text never contain `&`, `<`, or other
///   XML-escapable characters, so there is nothing to unescape, unlike
///   [`parse_spectrum_slice_quickxml`], which has to support it generally.
/// - **No namespace resolution, no attribute-order assumptions, no DOM.**
///   Just a forward-only scan for `<`, dispatch on the byte after it, and a
///   linear re-scan of a tag's (always short) attribute list per lookup.
///
/// Kept alongside the quick_xml path rather than replacing it --
/// `MzmlParserConfig::use_fast_tokenizer` selects between them, so a
/// real-world mzML variant this wasn't tested against can fall back to the
/// more general, better-established parser instead of silently mis-parsing.
pub fn parse_spectrum_slice_fast(
    slice: &[u8],
    id: usize,
    parse_metadata: bool,
) -> Option<MassSpectralRecord> {
    let len = slice.len();
    let mut pos = 0usize;

    let mut scan_number: Option<usize> = None;
    let mut precursor_mz: Option<f64> = None;
    let mut charge: i8 = 0;
    let mut polarity: i8 = 0;
    let mut isolation_target: Option<f64> = None;
    let mut isolation_lower: Option<f64> = None;
    let mut isolation_upper: Option<f64> = None;
    let mut collision_energy: Option<f64> = None;
    let mut retention_time_seconds: Option<f64> = None;
    let mut default_array_length: Option<usize> = None;
    // mzML `MS:1000511` ("ms level"); `MS:1000579` (MS1 spectrum) as a fallback.
    let mut ms_level: u8 = 0;

    let mut in_precursor = false;
    let mut current_array_kind = ArrayKind::Other;
    let mut current_precision = Precision::F64;
    let mut current_compressed = false;
    let mut current_numpress = Numpress::None;
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();

    while pos < len {
        let Some(rel_lt) = memchr::memchr(b'<', &slice[pos..]) else {
            // No more tags -- same as quick_xml's Event::Eof: stop and let
            // whatever peaks were already collected (if any) stand, rather
            // than treating a truncated tail as a hard parse failure.
            break;
        };
        let lt = pos + rel_lt;
        let after_lt = lt + 1;
        if after_lt >= len {
            break;
        }

        match slice[after_lt] {
            b'/' => {
                // End tag: </name>
                let Some(rel_gt) = memchr::memchr(b'>', &slice[after_lt..]) else {
                    break;
                };
                let gt = after_lt + rel_gt;
                let name = slice[after_lt + 1..gt].trim_ascii();
                if name == b"spectrum" {
                    break;
                } else if name == b"precursor" {
                    in_precursor = false;
                }
                pos = gt + 1;
            }
            b'?' | b'!' => {
                // Processing instruction / comment / CDATA -- not expected
                // inside a <spectrum> fragment, but skip to the next '>'
                // defensively rather than mis-tokenizing what follows.
                let Some(rel_gt) = memchr::memchr(b'>', &slice[after_lt..]) else {
                    break;
                };
                pos = after_lt + rel_gt + 1;
            }
            _ => {
                // Start or self-closing tag.
                let Some(rel_gt) = memchr::memchr(b'>', &slice[after_lt..]) else {
                    break;
                };
                let gt = after_lt + rel_gt;
                let self_closing = gt > after_lt && slice[gt - 1] == b'/';
                let tag_end = if self_closing { gt - 1 } else { gt };
                let tag_bytes = &slice[after_lt..tag_end];
                let name_end = tag_bytes
                    .iter()
                    .position(|b| b.is_ascii_whitespace())
                    .unwrap_or(tag_bytes.len());
                let name = &tag_bytes[..name_end];
                let attrs = &tag_bytes[name_end..];

                match name {
                    b"spectrum" if scan_number.is_none() => {
                        if let Some(v) = find_attr(attrs, b"id") {
                            if let Ok(s) = std::str::from_utf8(v) {
                                scan_number = extract_scan_number(s);
                            }
                        }
                        if let Some(v) = find_attr(attrs, b"defaultArrayLength") {
                            default_array_length =
                                std::str::from_utf8(v).ok().and_then(|s| s.parse().ok());
                        }
                    }
                    b"precursor" => in_precursor = true,
                    b"binaryDataArray" => {
                        current_array_kind = ArrayKind::Other;
                        current_precision = Precision::F64;
                        current_compressed = false;
                        current_numpress = Numpress::None;
                    }
                    b"cvParam" => {
                        if let Some(accession) = find_attr(attrs, b"accession") {
                            match accession {
                                // selected ion m/z
                                b"MS:1000130" => polarity = 1,
                                b"MS:1000129" => polarity = -1,
                                b"MS:1000827" if in_precursor => isolation_target = find_attr(attrs, b"value").and_then(|v| fast_float::parse::<f64, _>(v).ok()),
                                b"MS:1000828" if in_precursor => isolation_lower = find_attr(attrs, b"value").and_then(|v| fast_float::parse::<f64, _>(v).ok()),
                                b"MS:1000829" if in_precursor => isolation_upper = find_attr(attrs, b"value").and_then(|v| fast_float::parse::<f64, _>(v).ok()),
                                b"MS:1000045" if in_precursor => collision_energy = find_attr(attrs, b"value").and_then(|v| fast_float::parse::<f64, _>(v).ok()),
                                b"MS:1000744" if in_precursor => {
                                    precursor_mz = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.parse().ok());
                                }
                                // charge state
                                b"MS:1000041" if in_precursor => {
                                    charge = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .map(|v| v as i8)
                                        .unwrap_or(0);
                                }
                                // scan start time
                                b"MS:1000016" => {
                                    if let Some(t) = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.parse::<f64>().ok())
                                    {
                                        let is_minutes = find_attr(attrs, b"unitAccession")
                                            == Some(b"UO:0000031".as_slice());
                                        retention_time_seconds = Some(if is_minutes {
                                            t * 60.0 // minutes -> seconds
                                        } else {
                                            t // UO:0000010 (second), or unspecified: assume seconds
                                        });
                                    }
                                }
                                // ms level (outside <precursor>, at spectrum scope)
                                b"MS:1000511" if !in_precursor => {
                                    if let Some(l) = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.trim().parse::<u8>().ok())
                                    {
                                        ms_level = l;
                                    }
                                }
                                // "MS1 spectrum" boolean cvParam -- fallback when
                                // MS:1000511 is absent
                                b"MS:1000579" if ms_level == 0 => ms_level = 1,
                                // 64-bit float
                                b"MS:1000523" => current_precision = Precision::F64,
                                // 32-bit float
                                b"MS:1000521" => current_precision = Precision::F32,
                                // zlib compression
                                b"MS:1002745" => current_numpress = Numpress::Pic,
                                b"MS:1002747" => {
                                    current_numpress = Numpress::Pic;
                                    current_compressed = true;
                                }
                                b"MS:1002312" => current_numpress = Numpress::Linear,
                                b"MS:1002746" => {
                                    current_numpress = Numpress::Linear;
                                    current_compressed = true;
                                }
                                b"MS:1002314" => current_numpress = Numpress::Slof,
                                b"MS:1002748" => {
                                    current_numpress = Numpress::Slof;
                                    current_compressed = true;
                                }
                                b"MS:1000574" => current_compressed = true,
                                // no compression
                                b"MS:1000576" => current_compressed = false,
                                // m/z array
                                b"MS:1000514" => current_array_kind = ArrayKind::Mz,
                                // intensity array
                                b"MS:1000515" => current_array_kind = ArrayKind::Intensity,
                                _ => {}
                            }
                        }
                    }
                    b"binary" if !self_closing => {
                        let text_start = gt + 1;
                        let Some(rel_end) = memmem::find(&slice[text_start..], b"</binary>")
                        else {
                            // Unterminated <binary> -- malformed input, same
                            // treatment quick_xml gives a broken document.
                            return None;
                        };
                        let text_end = text_start + rel_end;
                        // Skip the decode entirely for an array kind we're
                        // not collecting -- some tools emit extra
                        // binaryDataArrays beyond m/z/intensity (e.g. ion
                        // mobility), and decoding+converting those just to
                        // discard the result would be pure waste.
                        if current_array_kind != ArrayKind::Other {
                            let text = &slice[text_start..text_end];
                            match current_array_kind {
                                ArrayKind::Mz => {
                                    if current_numpress != Numpress::None {
                                        if let Ok(raw) = decode_binary_bytes(
                                            text, current_compressed, Precision::F64, None,
                                        ) {
                                            mz_array = numpress_decode(current_numpress, &raw);
                                        }
                                    } else if let Ok(v) = decode_binary_f64(
                                        text,
                                        current_compressed,
                                        current_precision,
                                        default_array_length,
                                    ) {
                                        mz_array = v;
                                    }
                                }
                                ArrayKind::Intensity => {
                                    if current_numpress != Numpress::None {
                                        if let Ok(raw) = decode_binary_bytes(
                                            text, current_compressed, Precision::F32, None,
                                        ) {
                                            intensity_array = numpress::decode_pic(&raw)
                                                .into_iter()
                                                .map(|v| v as f32)
                                                .collect();
                                        }
                                    } else if let Ok(v) = decode_binary_f32(
                                        text,
                                        current_compressed,
                                        current_precision,
                                        default_array_length,
                                    ) {
                                        intensity_array = v;
                                    }
                                }
                                ArrayKind::Other => unreachable!(),
                            }
                        }
                        pos = text_end + b"</binary>".len();
                        continue;
                    }
                    _ => {}
                }
                pos = gt + 1;
            }
        }
    }

    let mut record = build_record(
        id,
        if polarity < 0 { -charge.abs() } else { charge },
        polarity,
        mz_array,
        intensity_array,
        parse_metadata,
        precursor_mz,
        retention_time_seconds,
        scan_number,
        if ms_level == 0 { 2 } else { ms_level },
    )?;
    if parse_metadata {
        if polarity != 0 { record.insert_metadata("IONMODE", if polarity < 0 { "Negative" } else { "Positive" }); }
        for (key, value) in [("ISOLATION_TARGET", isolation_target), ("ISOLATION_LOWER", isolation_lower),
                             ("ISOLATION_UPPER", isolation_upper), ("COLLISION_ENERGY", collision_energy)] {
            if let Some(v) = value { record.insert_metadata(key, &v.to_string()); }
        }
    }
    Some(record)
}

/// Scans `attrs` (the raw bytes after a tag's name, up to but not including
/// its closing `>`/`/>`) for `key="value"` or `key='value'`, returning the
/// value's raw bytes. Re-scans from the start on every call -- a tag's
/// attribute list is always short (a handful of entries), so this stays
/// cheap without needing to build any kind of map, the same tradeoff
/// `attr_value_f64`/`attr_value_eq` already make on the quick_xml path.
fn find_attr<'a>(attrs: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let n = attrs.len();
    let mut i = 0;
    while i < n {
        while i < n && attrs[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < n && attrs[i] != b'=' && !attrs[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_end = i;
        if key_start == key_end {
            break; // no more attributes
        }
        while i < n && attrs[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n || attrs[i] != b'=' {
            break; // malformed (attribute with no value) -- stop rather than misparse
        }
        i += 1;
        while i < n && attrs[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let quote = attrs[i];
        if quote != b'"' && quote != b'\'' {
            break; // malformed
        }
        i += 1;
        let val_start = i;
        while i < n && attrs[i] != quote {
            i += 1;
        }
        let val_end = i;
        if i < n {
            i += 1; // consume closing quote
        }
        if &attrs[key_start..key_end] == key {
            return Some(&attrs[val_start..val_end]);
        }
    }
    None
}

/// Parses `scan=N` out of an mzML spectrum `id` attribute (e.g.
/// `"controllerType=0 controllerNumber=1 scan=16574"`), matching how
/// `Incremental_Clustering`-style downstream tools key on scan number.
fn extract_scan_number(id_attr: &str) -> Option<usize> {
    id_attr.split_whitespace().find_map(|token| {
        token
            .strip_prefix("scan=")
            .and_then(|n| n.parse::<usize>().ok())
    })
}

/// Decodes a `<binary>` element's base64 text into a `Vec<f64>`, inflating
/// first if `compressed`. Values are read as `precision`-width
/// little-endian floats (mzML's on-disk byte order) and widened to `f64`
/// uniformly so callers don't need to care which precision the file used.
///
/// `text` is already-unescaped raw bytes: [`parse_spectrum_slice_quickxml`]
/// unescapes its `BytesText` before calling in (general XML text can contain
/// entity references); [`parse_spectrum_slice_fast`] passes its slice
/// straight from the document, since base64 text never contains `&`/`<` and
/// so never needs unescaping in the first place.
///
/// `expected_count`, when the spectrum's `defaultArrayLength` attribute was
/// present, lets the decompression buffer be sized exactly right up front
/// (`expected_count * bytes_per_value`) instead of guessing `raw.len() * 4`.
///
/// Returns raw little-endian value bytes, not `Vec<f64>`/`Vec<f32>` directly:
/// callers pick [`bytes_to_f64`] (m/z) or [`bytes_to_f32`] (intensity) so
/// each array is converted to its actual target type in one pass. An earlier
/// version always produced `Vec<f64>` here and had intensity's caller narrow
/// it to `f32` afterward in a second pass -- for `Precision::F32` source
/// data that meant widening f32->f64 here just to immediately narrow back
/// f64->f32 there, for no reason; for `Precision::F64` source data it meant
/// paying for a whole second allocation+pass that a single-pass conversion
/// doesn't need either way.
fn decode_binary_bytes(
    text: &[u8],
    compressed: bool,
    precision: Precision,
    expected_count: Option<usize>,
) -> Result<Vec<u8>, ()> {
    let trimmed = text.trim_ascii();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let raw = BASE64.decode(trimmed).map_err(|_| ())?;

    let bytes: Vec<u8> = if compressed {
        let bytes_per_value = match precision {
            Precision::F64 => 8,
            Precision::F32 => 4,
        };
        let capacity = expected_count
            .map(|n| n * bytes_per_value)
            .unwrap_or(raw.len() * 4);
        // libdeflate: profiling found it meaningfully faster than flate2's
        // zlib-ng backend for this workload (many small decompressions, not
        // one big one) -- purpose-built for decompression speed specifically,
        // where zlib-ng optimizes for both compress and decompress.
        let mut out: Vec<u8> = Vec::with_capacity(capacity.max(raw.len()));
        // SAFETY: `u8` has no validity invariant (every bit pattern is a
        // valid u8), so this doesn't expose any type-safety hole. The only
        // bytes ever read back out of `out` are `out[..n]` after
        // `zlib_decompress` reports it wrote exactly `n` bytes -- every byte
        // that's read was written first. Skips a full memset of a buffer
        // that's about to be entirely overwritten anyway, paid on every one
        // of ~2 binary arrays per spectrum, potentially megabytes each.
        unsafe {
            out.set_len(out.capacity());
        }
        let mut decompressor = libdeflater::Decompressor::new();
        loop {
            match decompressor.zlib_decompress(&raw, &mut out) {
                Ok(n) => {
                    out.truncate(n);
                    break;
                }
                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                    let additional = out.capacity().max(1);
                    out.reserve(additional);
                    let new_cap = out.capacity();
                    unsafe {
                        out.set_len(new_cap);
                    }
                }
                Err(_) => return Err(()),
            }
        }
        out
    } else {
        raw
    };

    Ok(bytes)
}

/// Converts raw little-endian value bytes to `Vec<f64>` in one pass --
/// widening is unavoidable for `Precision::F32` (the target type is wider
/// than the source), but at least it's one pass, not two.
fn bytes_to_f64(bytes: &[u8], precision: Precision) -> Vec<f64> {
    match precision {
        // SAFETY: chunks_exact(8) guarantees every yielded slice has length
        // exactly 8 (the remainder, if any, is never yielded), so try_into::<[u8; 8]>
        // cannot fail. Hot path (one call per peak value): skip the panic branch.
        Precision::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(unsafe { c.try_into().unwrap_unchecked() }))
            .collect(),
        Precision::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(unsafe { c.try_into().unwrap_unchecked() }) as f64)
            .collect(),
    }
}

/// Converts raw little-endian value bytes to `Vec<f32>` in one pass -- for
/// `Precision::F32` source data this is a direct byte read with no widening
/// ever involved, and for `Precision::F64` source data it narrows exactly
/// once, not widen-then-narrow via an intermediate `Vec<f64>`.
fn bytes_to_f32(bytes: &[u8], precision: Precision) -> Vec<f32> {
    match precision {
        // SAFETY: same chunks_exact(N) length guarantee as bytes_to_f64 above.
        Precision::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(unsafe { c.try_into().unwrap_unchecked() }) as f32)
            .collect(),
        Precision::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(unsafe { c.try_into().unwrap_unchecked() }))
            .collect(),
    }
}

/// Decodes a `<binary>` element straight into `Vec<T>` -- no separate
/// `Vec<u8>` byte buffer and no per-element `chunks_exact().map().collect()`
/// conversion pass at all, on the assumption that the file's on-disk
/// precision already matches `T`'s layout exactly (`x86_64`/`aarch64` are
/// both little-endian, matching mzML's on-disk byte order, so no byte
/// swapping is needed either). Only sound to call when that assumption
/// holds -- see `decode_binary_f64`/`decode_binary_f32`, which check
/// `precision` before calling this and fall back to the slower
/// convert-every-element path otherwise.
fn decode_binary_typed<T: bytemuck::Pod>(
    text: &[u8],
    compressed: bool,
    expected_count: Option<usize>,
) -> Result<Vec<T>, ()> {
    let trimmed = text.trim_ascii();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let element_size = std::mem::size_of::<T>();

    if !compressed {
        // Decode base64 straight into the final `Vec<T>` -- `out`'s
        // allocation is T-aligned by construction (it's a real `Vec<T>`, not
        // a byte buffer wearing a cast), so viewing it as `&mut [u8]` for
        // the decoder to write into is always sound, and there's no
        // separate `Vec<u8>` + copy/cast pass afterward at all, unlike the
        // compressed branch below (which genuinely needs an intermediate
        // `Vec<u8>` of *compressed* bytes for libdeflate to read from).
        let est = base64::decoded_len_estimate(trimmed.len());
        let mut out: Vec<T> = Vec::with_capacity(est.div_ceil(element_size));
        // SAFETY: `T: Pod` guarantees every bit pattern is a valid `T`; every
        // byte gets overwritten by `decode_slice` up to `n`, and `out` is
        // truncated to exactly `n / element_size` afterward.
        unsafe {
            out.set_len(out.capacity());
        }
        let out_bytes = bytemuck::cast_slice_mut::<T, u8>(&mut out);
        let n = BASE64.decode_slice(trimmed, out_bytes).map_err(|_| ())?;
        if n % element_size != 0 {
            return Err(());
        }
        out.truncate(n / element_size);
        return Ok(out);
    }

    let raw = BASE64.decode(trimmed).map_err(|_| ())?;
    let count_guess = expected_count.unwrap_or((raw.len() * 4) / element_size + 1);
    let mut out: Vec<T> = Vec::with_capacity(count_guess.max(raw.len() / element_size + 1));
    // SAFETY: `T: Pod` (bytemuck's bound) guarantees every bit pattern is a
    // valid `T` -- no padding, no invalid representations -- the same
    // reasoning `decode_binary_bytes`'s `Vec<u8>::set_len` relies on, just
    // generalized to `T`. Every element ever read back is one libdeflate
    // wrote first: after the loop, `out` is truncated to exactly
    // `n / element_size`, where `n` is the byte count libdeflate reports
    // having written.
    unsafe {
        out.set_len(out.capacity());
    }
    let mut decompressor = libdeflater::Decompressor::new();
    loop {
        let out_bytes = bytemuck::cast_slice_mut::<T, u8>(&mut out);
        match decompressor.zlib_decompress(&raw, out_bytes) {
            Ok(n) => {
                out.truncate(n / element_size);
                break;
            }
            Err(libdeflater::DecompressionError::InsufficientSpace) => {
                let additional = out.capacity().max(1);
                out.reserve(additional);
                let new_cap = out.capacity();
                unsafe {
                    out.set_len(new_cap);
                }
            }
            Err(_) => return Err(()),
        }
    }
    Ok(out)
}

/// Decodes a `<binary>` m/z array into `Vec<f64>`. When the source is
/// already `Precision::F64` (byte-identical layout to the target), this
/// decompresses straight into the final buffer via [`decode_binary_typed`],
/// skipping any per-element conversion. Only genuinely widens (via
/// [`bytes_to_f64`]) when the source is the narrower `Precision::F32`.
fn decode_binary_f64(
    text: &[u8],
    compressed: bool,
    precision: Precision,
    expected_count: Option<usize>,
) -> Result<Vec<f64>, ()> {
    match precision {
        Precision::F64 => decode_binary_typed::<f64>(text, compressed, expected_count),
        Precision::F32 => {
            let bytes = decode_binary_bytes(text, compressed, precision, expected_count)?;
            Ok(bytes_to_f64(&bytes, precision))
        }
    }
}

/// Decodes a `<binary>` intensity array into `Vec<f32>` -- mirrors
/// [`decode_binary_f64`]: direct via [`decode_binary_typed`] when the source
/// is already `Precision::F32`, narrowing via [`bytes_to_f32`] only when the
/// source is the wider `Precision::F64`.
fn decode_binary_f32(
    text: &[u8],
    compressed: bool,
    precision: Precision,
    expected_count: Option<usize>,
) -> Result<Vec<f32>, ()> {
    match precision {
        Precision::F32 => decode_binary_typed::<f32>(text, compressed, expected_count),
        Precision::F64 => {
            let bytes = decode_binary_bytes(text, compressed, precision, expected_count)?;
            Ok(bytes_to_f32(&bytes, precision))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a minimal-but-real one-spectrum mzML document with two
    /// zlib-compressed 64-bit-float binary arrays, so parsing can be
    /// checked against known peak values without any real instrument file.
    fn synthetic_spectrum_xml(
        scan: usize,
        precursor_mz: f64,
        charge: i8,
        rt_minutes: f64,
        mz: &[f64],
        intensity: &[f64],
    ) -> String {
        let encode_array = |values: &[f64]| -> String {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for &v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let mut compressed = Vec::new();
            {
                let mut encoder = flate2::write::ZlibEncoder::new(
                    &mut compressed,
                    flate2::Compression::default(),
                );
                encoder.write_all(&bytes).unwrap();
            }
            BASE64.encode(&compressed)
        };

        format!(
            r#"<spectrum id="controllerType=0 controllerNumber=1 scan={scan}" defaultArrayLength="{n}">
    <precursorList count="1">
      <precursor>
        <selectedIonList count="1">
          <selectedIon>
            <cvParam accession="MS:1000744" name="selected ion m/z" value="{precursor_mz}"/>
            <cvParam accession="MS:1000041" name="charge state" value="{charge}"/>
          </selectedIon>
        </selectedIonList>
      </precursor>
    </precursorList>
    <scanList count="1">
      <scan>
        <cvParam accession="MS:1000016" name="scan start time" value="{rt_minutes}" unitAccession="UO:0000031"/>
      </scan>
    </scanList>
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <cvParam accession="MS:1000574" name="zlib compression"/>
        <cvParam accession="MS:1000514" name="m/z array"/>
        <binary>{mz_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <cvParam accession="MS:1000574" name="zlib compression"/>
        <cvParam accession="MS:1000515" name="intensity array"/>
        <binary>{int_b64}</binary>
      </binaryDataArray>
    </binaryDataArrayList>
  </spectrum>"#,
            scan = scan,
            n = mz.len(),
            precursor_mz = precursor_mz,
            charge = charge,
            rt_minutes = rt_minutes,
            mz_b64 = encode_array(mz),
            int_b64 = encode_array(intensity),
        )
    }

    #[test]
    fn missing_charge_is_recovered_only_when_polarity_is_known() {
        let base = synthetic_spectrum_xml(7, 500.0, 1, 0.2, &[120.0, 250.0], &[10.0, 20.0])
            .replace(r#"<cvParam accession="MS:1000041" name="charge state" value="1"/>"#, "");
        assert!(!base.contains("MS:1000041"));
        let negative = base.replace("<precursorList", "<cvParam accession=\"MS:1000129\"/><precursorList");
        let positive = base.replace("<precursorList", "<cvParam accession=\"MS:1000130\"/><precursorList");
        for parser in [parse_spectrum_slice_fast, parse_spectrum_slice_quickxml] {
            for parse_metadata in [true, false] {
                assert_eq!(parser(negative.as_bytes(), 7, parse_metadata).unwrap().get_charge(), -1);
                assert_eq!(parser(positive.as_bytes(), 7, parse_metadata).unwrap().get_charge(), 1);
                assert_eq!(parser(base.as_bytes(), 7, parse_metadata).unwrap().get_charge(), 0);
            }
        }
    }

    #[test]
    fn both_parsers_preserve_negative_polarity_and_isolation_metadata() {
        let xml = synthetic_spectrum_xml(7, 500.0, 1, 0.2, &[120.0, 250.0], &[10.0, 20.0])
            .replace("<precursorList", "<cvParam accession=\"MS:1000129\"/><precursorList")
            .replace("<precursor>", "<precursor><isolationWindow><cvParam accession=\"MS:1000827\" value=\"500.0\"/><cvParam accession=\"MS:1000828\" value=\"0.4\"/><cvParam accession=\"MS:1000829\" value=\"0.6\"/></isolationWindow>");
        for parser in [parse_spectrum_slice_fast, parse_spectrum_slice_quickxml] {
            let record = parser(xml.as_bytes(), 7, true).unwrap();
            assert_eq!(record.get_charge(), -1);
            assert_eq!(record.get_metadata("IONMODE"), Some("Negative"));
            assert_eq!(record.get_metadata("ISOLATION_LOWER"), Some("0.4"));
            assert_eq!(record.get_metadata("ISOLATION_UPPER"), Some("0.6"));
        }
    }

    /// An MS1 survey scan: no `<precursor>`, an explicit `MS:1000511` level of 1.
    fn synthetic_ms1_spectrum_xml(
        scan: usize,
        rt_minutes: f64,
        mz: &[f64],
        intensity: &[f64],
    ) -> String {
        let encode_array = |values: &[f64]| -> String {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for &v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let mut compressed = Vec::new();
            {
                let mut encoder = flate2::write::ZlibEncoder::new(
                    &mut compressed,
                    flate2::Compression::default(),
                );
                encoder.write_all(&bytes).unwrap();
            }
            BASE64.encode(&compressed)
        };
        format!(
            r#"<spectrum id="controllerType=0 controllerNumber=1 scan={scan}" defaultArrayLength="{n}">
    <cvParam accession="MS:1000511" name="ms level" value="1"/>
    <cvParam accession="MS:1000579" name="MS1 spectrum" value=""/>
    <scanList count="1">
      <scan>
        <cvParam accession="MS:1000016" name="scan start time" value="{rt_minutes}" unitAccession="UO:0000031"/>
      </scan>
    </scanList>
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <cvParam accession="MS:1000574" name="zlib compression"/>
        <cvParam accession="MS:1000514" name="m/z array"/>
        <binary>{mz_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <cvParam accession="MS:1000574" name="zlib compression"/>
        <cvParam accession="MS:1000515" name="intensity array"/>
        <binary>{int_b64}</binary>
      </binaryDataArray>
    </binaryDataArrayList>
  </spectrum>"#,
            scan = scan,
            n = mz.len(),
            rt_minutes = rt_minutes,
            mz_b64 = encode_array(mz),
            int_b64 = encode_array(intensity),
        )
    }

    fn wrap_indexed_mzml(spectra_xml: &[String]) -> Vec<u8> {
        let header = format!(
            r#"<?xml version="1.0" encoding="ISO-8859-1"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML>
  <run id="synthetic">
    <spectrumList count="{count}">
"#,
            count = spectra_xml.len()
        );
        let mut doc = header.into_bytes();
        let mut spectrum_offsets = Vec::new();
        for s in spectra_xml {
            spectrum_offsets.push(doc.len());
            doc.extend_from_slice(s.as_bytes());
            doc.push(b'\n');
        }
        doc.extend_from_slice(b"    </spectrumList>\n  </run>\n</mzML>\n");

        let index_list_offset = doc.len();
        doc.extend_from_slice(b"<indexList count=\"1\">\n  <index name=\"spectrum\">\n");
        for &off in &spectrum_offsets {
            doc.extend_from_slice(format!("    <offset idRef=\"x\">{off}</offset>\n").as_bytes());
        }
        doc.extend_from_slice(b"  </index>\n</indexList>\n");
        doc.extend_from_slice(
            format!("<indexListOffset>{index_list_offset}</indexListOffset>\n").as_bytes(),
        );
        doc.extend_from_slice(b"<fileChecksum>0</fileChecksum>\n</indexedmzML>\n");
        doc
    }

    fn wrap_plain_mzml(spectra_xml: &[String]) -> Vec<u8> {
        let mut doc = format!(
            r#"<?xml version="1.0" encoding="ISO-8859-1"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml">
  <run id="synthetic">
    <spectrumList count="{count}">
"#,
            count = spectra_xml.len()
        )
        .into_bytes();
        for s in spectra_xml {
            doc.extend_from_slice(s.as_bytes());
            doc.push(b'\n');
        }
        doc.extend_from_slice(b"    </spectrumList>\n  </run>\n</mzML>\n");
        doc
    }

    #[test]
    fn indexed_synthetic_file_parses_correctly() {
        let s1 = synthetic_spectrum_xml(101, 500.25, 2, 12.5, &[100.1, 200.2, 300.3], &[10.0, 999.0, 50.0]);
        let s2 = synthetic_spectrum_xml(102, 611.30, 1, 13.0, &[150.5, 250.6], &[20.0, 30.0]);
        let doc = wrap_indexed_mzml(&[s1, s2]);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxidems_test_indexed_{}.mzML", std::process::id()));
        std::fs::write(&path, &doc).unwrap();

        let mut parser = MzmlParser::default();
        let records = parser.parse_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].peaks().len(), 3);
        assert_eq!(records[1].peaks().len(), 2);
        assert!((records[0].peaks()[0].mz - 100.1).abs() < 1e-9);
        assert!((records[0].peaks()[1].intensity - 999.0).abs() < 1e-3);
        assert_eq!(records[0].get_charge(), 2);
        assert_eq!(records[0].get_metadata("PEPMASS"), Some("500.25"));
        assert_eq!(records[0].get_metadata("SCANS"), Some("101"));
        // 12.5 minutes -> 750 seconds
        assert_eq!(records[0].get_metadata("RTINSECONDS"), Some("750"));
    }

    #[test]
    fn ms1_survey_scans_are_parsed_with_level_1() {
        let ms1 = synthetic_ms1_spectrum_xml(1, 5.0, &[300.0, 301.0, 302.0], &[100.0, 6.0, 1.0]);
        let ms2a = synthetic_spectrum_xml(2, 300.10, 1, 5.01, &[80.0, 120.0], &[5.0, 9.0]);
        let ms2b = synthetic_spectrum_xml(3, 450.20, 1, 5.20, &[90.0, 200.0], &[4.0, 7.0]);
        let doc = wrap_indexed_mzml(&[ms1, ms2a, ms2b]);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxidems_test_ms1_{}.mzML", std::process::id()));
        std::fs::write(&path, &doc).unwrap();
        let mut parser = MzmlParser::default();
        let records = parser.parse_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].ms_level, 1);
        assert_eq!(records[1].ms_level, 2);
        assert_eq!(records[2].ms_level, 2);
        // MS1 scan still carries its retention time (5 min -> 300 s).
        assert_eq!(records[0].get_metadata("RTINSECONDS"), Some("300"));
        assert_eq!(records[0].get_metadata("PEPMASS"), None);
        assert_eq!(records[0].peaks().len(), 3);
    }

    #[test]
    fn scanned_fallback_matches_indexed_path() {
        let s1 = synthetic_spectrum_xml(1, 400.0, 1, 5.0, &[110.0, 220.0], &[1.0, 2.0]);
        let s2 = synthetic_spectrum_xml(2, 405.0, 1, 5.1, &[111.0, 221.0, 331.0], &[3.0, 4.0, 5.0]);
        let doc = wrap_plain_mzml(&[s1, s2]);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxidems_test_plain_{}.mzML", std::process::id()));
        std::fs::write(&path, &doc).unwrap();

        // Confirm this file genuinely has no usable index (exercises the
        // fallback scan path, not just "happens to still work").
        assert!(MzmlParser::indexed_spans(&doc).is_none());

        let mut parser = MzmlParser::default();
        let records = parser.parse_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].peaks().len(), 2);
        assert_eq!(records[1].peaks().len(), 3);
    }

    #[test]
    fn uncompressed_32bit_arrays_parse_correctly() {
        // Same shape as synthetic_spectrum_xml but without zlib compression
        // and with 32-bit floats, to exercise that branch combination too.
        let mz = [123.456f32, 234.567, 345.678];
        let intensity = [1.0f32, 2.0, 3.0];
        let mut mz_bytes = Vec::new();
        for v in mz {
            mz_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut int_bytes = Vec::new();
        for v in intensity {
            int_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let spectrum = format!(
            r#"<spectrum id="scan=7" defaultArrayLength="3">
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000521" name="32-bit float"/>
        <cvParam accession="MS:1000576" name="no compression"/>
        <cvParam accession="MS:1000514" name="m/z array"/>
        <binary>{mz_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000521" name="32-bit float"/>
        <cvParam accession="MS:1000576" name="no compression"/>
        <cvParam accession="MS:1000515" name="intensity array"/>
        <binary>{int_b64}</binary>
      </binaryDataArray>
    </binaryDataArrayList>
  </spectrum>"#,
            mz_b64 = BASE64.encode(&mz_bytes),
            int_b64 = BASE64.encode(&int_bytes),
        );
        let doc = wrap_plain_mzml(&[spectrum]);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxidems_test_32bit_{}.mzML", std::process::id()));
        std::fs::write(&path, &doc).unwrap();
        let mut parser = MzmlParser::default();
        let records = parser.parse_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].peaks().len(), 3);
        assert!((records[0].peaks()[0].mz - 123.456).abs() < 1e-3);
        assert!((records[0].peaks()[2].intensity - 3.0).abs() < 1e-4);
    }

    #[test]
    fn spectrum_with_no_peaks_is_dropped() {
        let empty = synthetic_spectrum_xml(1, 100.0, 1, 1.0, &[], &[]);
        let doc = wrap_plain_mzml(&[empty]);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxidems_test_empty_{}.mzML", std::process::id()));
        std::fs::write(&path, &doc).unwrap();
        let mut parser = MzmlParser::default();
        let records = parser.parse_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(records.len(), 0);
    }

    /// Compares a `Vec<MassSpectralRecord>` field-by-field -- used to check
    /// the fast and quick_xml tokenizers agree exactly, not just "both
    /// produced some records".
    fn assert_records_equivalent(fast: &[MassSpectralRecord], quickxml: &[MassSpectralRecord]) {
        assert_eq!(fast.len(), quickxml.len(), "record count differs");
        for (a, b) in fast.iter().zip(quickxml.iter()) {
            assert_eq!(a.get_charge(), b.get_charge());
            assert_eq!(a.get_metadata("PEPMASS"), b.get_metadata("PEPMASS"));
            assert_eq!(a.get_metadata("RTINSECONDS"), b.get_metadata("RTINSECONDS"));
            assert_eq!(a.get_metadata("SCANS"), b.get_metadata("SCANS"));
            assert_eq!(a.peaks().len(), b.peaks().len());
            for (pa, pb) in a.peaks().iter().zip(b.peaks().iter()) {
                assert!((pa.mz - pb.mz).abs() < 1e-9, "mz mismatch: {} vs {}", pa.mz, pb.mz);
                assert!(
                    (pa.intensity - pb.intensity).abs() < 1e-4,
                    "intensity mismatch: {} vs {}",
                    pa.intensity,
                    pb.intensity
                );
            }
        }
    }

    /// The fast hand-rolled tokenizer and the quick_xml fallback must agree
    /// exactly on every synthetic fixture used elsewhere in this file --
    /// indexed, non-indexed, 32-bit uncompressed, multi-spectrum. Both
    /// `use_fast_tokenizer: true/false` are real, user-selectable configs
    /// (see `MzmlParserConfig`), so both need their own passing coverage,
    /// not just "the default happens to work".
    #[test]
    fn fast_and_quickxml_tokenizers_agree_on_synthetic_files() {
        let s1 = synthetic_spectrum_xml(101, 500.25, 2, 12.5, &[100.1, 200.2, 300.3], &[10.0, 999.0, 50.0]);
        let s2 = synthetic_spectrum_xml(102, 611.30, 1, 13.0, &[150.5, 250.6], &[20.0, 30.0]);
        let doc = wrap_indexed_mzml(&[s1, s2]);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxidems_test_tokenizer_cmp_{}.mzML", std::process::id()));
        std::fs::write(&path, &doc).unwrap();

        let mut fast_parser = MzmlParser::new(MzmlParserConfig {
            use_fast_tokenizer: true,
            ..MzmlParserConfig::default()
        });
        let fast_records = fast_parser.parse_file(path.to_str().unwrap()).unwrap();

        let mut quickxml_parser = MzmlParser::new(MzmlParserConfig {
            use_fast_tokenizer: false,
            ..MzmlParserConfig::default()
        });
        let quickxml_records = quickxml_parser.parse_file(path.to_str().unwrap()).unwrap();

        std::fs::remove_file(&path).ok();

        assert_eq!(fast_records.len(), 2);
        assert_records_equivalent(&fast_records, &quickxml_records);
    }

    /// Cross-validates both boundary-finding paths against a real
    /// instrument-derived indexed mzML file, not just synthetic documents:
    /// the indexed fast path and the memchr fallback scan must agree on
    /// every span, spectrum-for-spectrum, on real data with real XML
    /// irregularities (whitespace, attribute ordering, etc. synthetic test
    /// fixtures don't naturally exercise). Gated behind `--ignored` since it
    /// depends on OXIDEMS_TEST_MZML_FIXTURE pointing at a real mzML file, not
    /// something every clone of oxidems_lib will have on disk.
    #[test]
    #[ignore = "depends on OXIDEMS_TEST_MZML_FIXTURE; run explicitly with --ignored"]
    fn indexed_and_scanned_paths_agree_on_real_file() {
        let path = std::env::var("OXIDEMS_TEST_MZML_FIXTURE")
            .expect("set OXIDEMS_TEST_MZML_FIXTURE to a real mzML file to run this ignored test");
        let file = std::fs::File::open(&path).expect("real test fixture missing");
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };

        let indexed = MzmlParser::indexed_spans(&mmap).expect("file should carry a valid index");
        let scanned = MzmlParser::scanned_spans(&mmap);

        assert_eq!(indexed.len(), scanned.len(), "span count must match");
        assert_eq!(indexed, scanned, "every span must match exactly");

        // And both should agree with independent ground truth: a plain
        // substring count of "<spectrum " occurrences.
        let expected = memchr::memmem::find_iter(&mmap[..], b"<spectrum ").count();
        assert_eq!(indexed.len(), expected);

        let mut parser = MzmlParser::default();
        let records = parser.parse_file(&path).unwrap();
        assert_eq!(records.len(), expected);
        assert!(records.iter().all(|r| !r.peaks().is_empty()));
        assert!(records.iter().all(|r| r.get_metadata("PEPMASS").is_some()));
    }

    /// The real cross-check for the hand-rolled tokenizer: does it agree
    /// with the established quick_xml path on an actual instrument-derived
    /// file, with all the real-world XML irregularities (whitespace,
    /// attribute ordering, spectra with no charge/precursor cvParam, etc.)
    /// synthetic fixtures don't naturally produce. Gated behind `--ignored`
    /// for the same reason as `indexed_and_scanned_paths_agree_on_real_file`.
    #[test]
    #[ignore = "depends on OXIDEMS_TEST_MZML_FIXTURE; run explicitly with --ignored"]
    fn fast_and_quickxml_tokenizers_agree_on_real_file() {
        let path = std::env::var("OXIDEMS_TEST_MZML_FIXTURE")
            .expect("set OXIDEMS_TEST_MZML_FIXTURE to a real mzML file to run this ignored test");

        let mut fast_parser = MzmlParser::new(MzmlParserConfig {
            use_fast_tokenizer: true,
            ..MzmlParserConfig::default()
        });
        let fast_records = fast_parser.parse_file(&path).unwrap();

        let mut quickxml_parser = MzmlParser::new(MzmlParserConfig {
            use_fast_tokenizer: false,
            ..MzmlParserConfig::default()
        });
        let quickxml_records = quickxml_parser.parse_file(&path).unwrap();

        assert_records_equivalent(&fast_records, &quickxml_records);
    }
}

/// MS-Numpress "positive integer" (PIC) decoding, `MS:1002746`/`MS:1002747`.
///
/// ProteoWizard writes intensity arrays this way by default for Thermo data, and
/// a reader that does not know it decodes the payload as raw floats, produces an
/// array of the wrong length, and drops the spectrum. Every MSnLib file was
/// unreadable for exactly that reason.
///
/// PIC packs each value as a nibble stream: a head nibble giving the count of
/// leading zero nibbles (`<= 8`), or `8 + count` for leading *ones*, followed by
/// the remaining nibbles of the 32-bit integer, least-significant first.
/// Reference: Teleman et al., *Mol Cell Proteomics* 2014, `MSNumpress.cpp`.
/// Applies whichever MS-Numpress transform an array declared.
fn numpress_decode(kind: Numpress, raw: &[u8]) -> Vec<f64> {
    match kind {
        Numpress::None => Vec::new(),
        Numpress::Pic => numpress::decode_pic(raw),
        Numpress::Linear => numpress::decode_linear(raw),
        Numpress::Slof => numpress::decode_slof(raw),
    }
}

/// Which MS-Numpress transform a binary array carries, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numpress {
    None,
    /// Positive-integer compression (`MS:1002745`, or `MS:1002747` with zlib).
    Pic,
    /// Linear prediction (`MS:1002312`, or `MS:1002746` with zlib).
    Linear,
    /// Short logged float (`MS:1002314`, or `MS:1002748` with zlib).
    Slof,
}

mod numpress {
    /// Reads one nibble, advancing the byte cursor every second read.
    #[inline]
    fn nibble(data: &[u8], di: &mut usize, half: &mut bool) -> Option<u8> {
        let b = *data.get(*di)?;
        let v = if !*half { b >> 4 } else { b & 0x0f };
        if *half {
            *di += 1;
        }
        *half = !*half;
        Some(v)
    }

    fn decode_int(data: &[u8], di: &mut usize, half: &mut bool) -> Option<u32> {
        let head = nibble(data, di, half)?;
        let (n, mut res) = if head <= 8 {
            (head as usize, 0u32)
        } else {
            // Leading ones: the top `head - 8` nibbles are all 1s.
            let n = (head - 8) as usize;
            let mut r = 0u32;
            for i in 0..n {
                r |= 0xF << (4 * (7 - i));
            }
            (n, r)
        };
        if n == 8 {
            return Some(res);
        }
        for j in 0..(8 - n) {
            let nib = nibble(data, di, half)? as u32;
            res |= nib << (4 * j);
        }
        Some(res)
    }

    /// Linear prediction (`numLinear`). The payload opens with an 8-byte fixed-point
    /// scaling factor and two seed values; every later value is encoded as the
    /// residual against the linear extrapolation of the previous two.
    pub fn decode_linear(data: &[u8]) -> Vec<f64> {
        if data.len() < 8 {
            return Vec::new();
        }
        let mut fixed = 0f64;
        for i in 0..8 {
            fixed += (data[i] as f64) * 2f64.powi(8 * i as i32);
        }
        if fixed == 0.0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        if data.len() < 12 {
            return out;
        }
        let first = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as f64;
        out.push(first / fixed);
        if data.len() < 16 {
            return out;
        }
        let second = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as f64;
        out.push(second / fixed);
        let (mut di, mut half) = (16usize, false);
        while di < data.len() {
            if di == data.len() - 1 && half && (data[di] & 0x0f) == 0 {
                break;
            }
            let Some(diff) = decode_int(data, &mut di, &mut half) else { break };
            let n = out.len();
            // Two previous values, in fixed point, extrapolated linearly.
            let a = (out[n - 1] * fixed).round();
            let b = (out[n - 2] * fixed).round();
            let pred = 2.0 * a - b;
            let val = pred + (diff as i32) as f64;
            out.push(val / fixed);
        }
        out
    }

    /// Short logged float (`numSlof`): each value is `ln(v+1)` scaled into a u16.
    pub fn decode_slof(data: &[u8]) -> Vec<f64> {
        if data.len() < 8 {
            return Vec::new();
        }
        let mut fixed = 0f64;
        for i in 0..8 {
            fixed += (data[i] as f64) * 2f64.powi(8 * i as i32);
        }
        if fixed == 0.0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity((data.len() - 8) / 2);
        let mut i = 8;
        while i + 1 < data.len() {
            let x = u16::from_le_bytes([data[i], data[i + 1]]) as f64;
            out.push((x / fixed).exp() - 1.0);
            i += 2;
        }
        out
    }

    /// Decodes a whole PIC payload (already zlib-inflated) into f64 values.
    pub fn decode_pic(data: &[u8]) -> Vec<f64> {
        let mut out = Vec::new();
        let (mut di, mut half) = (0usize, false);
        while di < data.len() {
            // A final half-consumed byte whose low nibble is zero is padding.
            if di == data.len() - 1 && half && (data[di] & 0x0f) == 0 {
                break;
            }
            match decode_int(data, &mut di, &mut half) {
                Some(v) => out.push(v as f64),
                None => break,
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Mirror of the reference `encodeInt`, so the decoder is tested against an
        /// independent implementation of the format rather than against itself.
        fn encode_int(x: u32, out: &mut Vec<u8>) {
            let mut nibbles = Vec::new();
            let init = x & 0xf000_0000;
            let l = if init == 0 {
                (0..8).find(|i| (x & (0xf000_0000u32 >> (4 * i))) != 0).unwrap_or(8)
            } else if init == 0xf000_0000 {
                (0..8)
                    .find(|i| {
                        let m = 0xf000_0000u32 >> (4 * i);
                        (x & m) != m
                    })
                    .unwrap_or(8)
            } else {
                0
            };
            let head = if init == 0xf000_0000 { l + 8 } else { l };
            nibbles.push(head as u8);
            for j in 0..(8 - l) {
                nibbles.push(((x >> (4 * j)) & 0xf) as u8);
            }
            out.extend(nibbles);
        }

        fn pack(nibbles: &[u8]) -> Vec<u8> {
            let mut b = Vec::new();
            for pair in nibbles.chunks(2) {
                let hi = pair[0] << 4;
                let lo = if pair.len() > 1 { pair[1] } else { 0 };
                b.push(hi | lo);
            }
            b
        }

        /// Slof is exactly invertible on its own definition, so encode it here and
        /// require the decoder to recover the values within the format's precision.
        #[test]
        fn slof_round_trips() {
            let vals = [0.0f64, 1.0, 10.0, 1234.0, 8_483_545.0];
            let max = vals.iter().cloned().fold(0.0f64, f64::max);
            // The 8-byte header is a byte-decomposed integer, so a fractional
            // scaling factor would not survive it — encoder and decoder would then
            // disagree on the scale and every value would drift. Round it, as the
            // reference encoder does.
            let fixed = ((0xFFFFu32 as f64) / (max + 1.0).ln()).floor();
            let mut data = Vec::new();
            // 8-byte little-endian fixed-point header, as `numSlof` writes it.
            let mut f = fixed;
            for _ in 0..8 {
                data.push((f as u64 & 0xff) as u8);
                f /= 256.0;
            }
            for &v in &vals {
                let x = ((v + 1.0).ln() * fixed).round() as u16;
                data.extend_from_slice(&x.to_le_bytes());
            }
            let got = decode_slof(&data);
            assert_eq!(got.len(), vals.len());
            for (g, w) in got.iter().zip(vals.iter()) {
                // 16 bits over a log scale: relative, not absolute, precision.
                assert!((g - w).abs() <= (w.abs() * 1e-3).max(1e-3), "got {g} want {w}");
            }
        }

        /// Linear must at minimum recover its two seed values exactly; those are
        /// stored as plain fixed-point and are what a truncated payload yields.
        #[test]
        fn linear_recovers_its_seed_values() {
            let fixed = 1000.0f64;
            let mut data = Vec::new();
            let mut f = fixed;
            for _ in 0..8 {
                data.push((f as u64 & 0xff) as u8);
                f /= 256.0;
            }
            data.extend_from_slice(&((100.5 * fixed) as u32).to_le_bytes());
            data.extend_from_slice(&((200.25 * fixed) as u32).to_le_bytes());
            let got = decode_linear(&data);
            assert_eq!(got.len(), 2);
            assert!((got[0] - 100.5).abs() < 1e-6);
            assert!((got[1] - 200.25).abs() < 1e-6);
        }

        #[test]
        fn pic_round_trips_through_an_independent_encoder() {
            let vals: Vec<u32> = vec![0, 1, 15, 16, 255, 4096, 65535, 1_000_000, 8_483_545];
            let mut nib = Vec::new();
            for &v in &vals {
                encode_int(v, &mut nib);
            }
            let bytes = pack(&nib);
            let got = decode_pic(&bytes);
            let want: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
            assert_eq!(&got[..want.len()], &want[..]);
        }
    }
}
