use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use memchr::memmem;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io;

use crate::io::mzml::MzmlParser;
use crate::spectrum::MassSpectralRecord;

/// Lightweight precursor metadata for MS2 fragmentation scans.
///
/// Avoids decoding or storing MS2 peak arrays when only precursor m/z and retention time
/// are required for feature linking.
#[derive(Debug, Clone)]
pub struct Ms2PrecursorInfo {
    pub scan_id: usize,
    pub precursor_mz: f64,
    pub rt: f32,
    pub charge: i8,
}

/// Contiguous Struct-of-Arrays (SoA) representation of an MS1 LC-MS acquisition run.
///
/// Flattening all MS1 scans into continuous slices of `m/z` and `intensity` provides
/// cache locality, eliminates per-scan vector allocations, and enables SIMD/parallel sweeps.
#[derive(Debug, Clone, Default)]
pub struct Ms1Run {
    /// Retention time (in seconds) for each scan.
    pub scan_rts: Vec<f32>,
    /// Native scan numbers from source file (if available).
    pub scan_numbers: Vec<usize>,
    /// Slice offsets into `mz` and `intensity`. Length is `scan_rts.len() + 1`.
    /// Scan `i` spans indices `scan_offsets[i]..scan_offsets[i+1]`.
    pub scan_offsets: Vec<usize>,
    /// Flat array of m/z values, sorted ascending within each scan.
    pub mz: Vec<f64>,
    /// Flat array of corresponding peak intensities.
    pub intensity: Vec<f32>,
}

impl Ms1Run {
    /// Creates an `Ms1Run` by collecting and filtering MS1 records from parsed spectra.
    pub fn from_records(records: &[MassSpectralRecord]) -> Self {
        let mut scan_rts = Vec::new();
        let mut scan_numbers = Vec::new();
        let mut scan_offsets = Vec::with_capacity(records.len() + 1);
        scan_offsets.push(0);

        let mut mz = Vec::new();
        let mut intensity = Vec::new();

        for record in records {
            // Only process MS1 survey scans
            if record.ms_level != 1 {
                continue;
            }

            let rt = record
                .get_metadata("RTINSECONDS")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0);

            let scan_no = record
                .get_metadata("SCANS")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(scan_rts.len() + 1);

            let peaks = record.peaks();
            scan_rts.push(rt);
            scan_numbers.push(scan_no);

            // Verify if peaks are sorted; if not, sort them
            let is_sorted = peaks.windows(2).all(|w| w[0].mz <= w[1].mz);
            if is_sorted {
                for p in peaks {
                    mz.push(p.mz);
                    intensity.push(p.intensity);
                }
            } else {
                let mut sorted_peaks = peaks.to_vec();
                sorted_peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
                for p in &sorted_peaks {
                    mz.push(p.mz);
                    intensity.push(p.intensity);
                }
            }

            scan_offsets.push(mz.len());
        }

        Self {
            scan_rts,
            scan_numbers,
            scan_offsets,
            mz,
            intensity,
        }
    }

    /// Ultra-fast selective ingestion directly from an mzML file.
    ///
    /// Directly streams MS1 binary peak arrays into flat contiguous memory while
    /// skipping binary decoding for MS2 fragmentation scans (only extracting precursor m/z,
    /// RT, and charge). Bypasses intermediate `Vec<Peak>` and string metadata allocations.
    pub fn from_mzml_file_direct(path: &str) -> io::Result<(Self, Vec<Ms2PrecursorInfo>)> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let spans = MzmlParser::spectrum_spans(&mmap);

        let results: Vec<ParsedSpectrumItem> = spans
            .par_iter()
            .enumerate()
            .filter_map(|(i, &(start, end))| parse_spectrum_selective(&mmap[start..end], i + 1))
            .collect();

        let mut scan_rts = Vec::new();
        let mut scan_numbers = Vec::new();
        let mut scan_offsets = Vec::new();
        scan_offsets.push(0);
        let mut mz = Vec::new();
        let mut intensity = Vec::new();
        let mut ms2_info = Vec::new();

        for item in results {
            match item {
                ParsedSpectrumItem::Ms1 { scan_no, rt, mut mzs, mut ints } => {
                    scan_rts.push(rt);
                    scan_numbers.push(scan_no);

                    let is_sorted = mzs.windows(2).all(|w| w[0] <= w[1]);
                    if !is_sorted {
                        let mut pairs: Vec<(f64, f32)> = mzs.into_iter().zip(ints).collect();
                        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                        for (m, int) in pairs {
                            mz.push(m);
                            intensity.push(int);
                        }
                    } else {
                        mz.extend(mzs);
                        intensity.extend(ints);
                    }
                    scan_offsets.push(mz.len());
                }
                ParsedSpectrumItem::Ms2(info) => {
                    ms2_info.push(info);
                }
            }
        }

        let run = Self {
            scan_rts,
            scan_numbers,
            scan_offsets,
            mz,
            intensity,
        };

        Ok((run, ms2_info))
    }

    /// Number of MS1 scans.
    #[inline]
    pub fn num_scans(&self) -> usize {
        self.scan_rts.len()
    }

    /// Total number of MS1 centroid peaks across all scans.
    #[inline]
    pub fn total_peaks(&self) -> usize {
        self.mz.len()
    }

    /// Returns the slices `(&[mz], &[intensity])` for a given scan index.
    #[inline]
    pub fn scan_peaks(&self, scan_idx: usize) -> (&[f64], &[f32]) {
        let start = self.scan_offsets[scan_idx];
        let end = self.scan_offsets[scan_idx + 1];
        (&self.mz[start..end], &self.intensity[start..end])
    }

    /// Returns the retention time (seconds) of scan `scan_idx`.
    #[inline]
    pub fn scan_rt(&self, scan_idx: usize) -> f32 {
        self.scan_rts[scan_idx]
    }

    /// Returns `(min_rt, max_rt)` in seconds.
    pub fn rt_range(&self) -> (f32, f32) {
        if self.scan_rts.is_empty() {
            return (0.0, 0.0);
        }
        let min_rt = *self.scan_rts.first().unwrap();
        let max_rt = *self.scan_rts.last().unwrap();
        (min_rt, max_rt)
    }

    /// Returns `(min_mz, max_mz)` across all scans.
    pub fn mz_range(&self) -> (f64, f64) {
        if self.mz.is_empty() {
            return (0.0, 0.0);
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &m in &self.mz {
            if m < min {
                min = m;
            }
            if m > max {
                max = m;
            }
        }
        (min, max)
    }
}

enum ParsedSpectrumItem {
    Ms1 {
        scan_no: usize,
        rt: f32,
        mzs: Vec<f64>,
        ints: Vec<f32>,
    },
    Ms2(Ms2PrecursorInfo),
}

#[derive(PartialEq)]
enum ArrayKind {
    Mz,
    Intensity,
    Other,
}

/// Hand-rolled selective parser that skips `<binary>` decoding for MS2 scans.
fn parse_spectrum_selective(slice: &[u8], default_id: usize) -> Option<ParsedSpectrumItem> {
    let len = slice.len();
    let mut pos = 0usize;

    let mut scan_number: Option<usize> = None;
    let mut precursor_mz: Option<f64> = None;
    let mut charge: i8 = 0;
    let mut rt_sec: Option<f32> = None;
    let mut ms_level: u8 = 0;
    let mut in_precursor = false;

    let mut current_array_kind = ArrayKind::Other;
    let mut current_compressed = false;
    let mut current_f32 = false;
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();

    while pos < len {
        let Some(rel_lt) = memchr::memchr(b'<', &slice[pos..]) else { break; };
        let lt = pos + rel_lt;
        let after_lt = lt + 1;
        if after_lt >= len { break; }

        match slice[after_lt] {
            b'/' => {
                let Some(rel_gt) = memchr::memchr(b'>', &slice[after_lt..]) else { break; };
                let gt = after_lt + rel_gt;
                let name = slice[after_lt + 1..gt].trim_ascii();
                if name == b"spectrum" { break; }
                else if name == b"precursor" { in_precursor = false; }
                pos = gt + 1;
            }
            b'?' | b'!' => {
                let Some(rel_gt) = memchr::memchr(b'>', &slice[after_lt..]) else { break; };
                pos = after_lt + rel_gt + 1;
            }
            _ => {
                let Some(rel_gt) = memchr::memchr(b'>', &slice[after_lt..]) else { break; };
                let gt = after_lt + rel_gt;
                let self_closing = gt > after_lt && slice[gt - 1] == b'/';
                let tag_end = if self_closing { gt - 1 } else { gt };
                let tag_bytes = &slice[after_lt..tag_end];
                let name_end = tag_bytes.iter().position(|b| b.is_ascii_whitespace()).unwrap_or(tag_bytes.len());
                let name = &tag_bytes[..name_end];
                let attrs = &tag_bytes[name_end..];

                match name {
                    b"spectrum" if scan_number.is_none() => {
                        if let Some(v) = find_attr(attrs, b"id") {
                            if let Ok(s) = std::str::from_utf8(v) {
                                scan_number = s.split_whitespace().find_map(|t| t.strip_prefix("scan=").and_then(|n| n.parse::<usize>().ok()));
                            }
                        }
                    }
                    b"precursor" => in_precursor = true,
                    b"binaryDataArrayList" | b"binaryDataArray" => {
                        // Crucial optimization: If MS2 scan, we already extracted precursor_mz and rt!
                        // Skip binary peak arrays completely.
                        if ms_level >= 2 {
                            if let Some(pmz) = precursor_mz {
                                return Some(ParsedSpectrumItem::Ms2(Ms2PrecursorInfo {
                                    scan_id: scan_number.unwrap_or(default_id),
                                    precursor_mz: pmz,
                                    rt: rt_sec.unwrap_or(0.0),
                                    charge,
                                }));
                            }
                            return None;
                        }
                        current_array_kind = ArrayKind::Other;
                        current_compressed = false;
                        current_f32 = false;
                    }
                    b"cvParam" => {
                        if let Some(accession) = find_attr(attrs, b"accession") {
                            match accession {
                                b"MS:1000744" if in_precursor => {
                                    precursor_mz = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.parse().ok());
                                }
                                b"MS:1000041" if in_precursor => {
                                    charge = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .map(|v| v as i8).unwrap_or(0);
                                }
                                b"MS:1000016" => {
                                    if let Some(t) = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.parse::<f64>().ok())
                                    {
                                        let is_min = find_attr(attrs, b"unitAccession") == Some(b"UO:0000031");
                                        rt_sec = Some(if is_min { (t * 60.0) as f32 } else { t as f32 });
                                    }
                                }
                                b"MS:1000511" if !in_precursor => {
                                    if let Some(l) = find_attr(attrs, b"value")
                                        .and_then(|v| std::str::from_utf8(v).ok())
                                        .and_then(|s| s.trim().parse::<u8>().ok())
                                    {
                                        ms_level = l;
                                    }
                                }
                                b"MS:1000579" if ms_level == 0 => ms_level = 1,
                                b"MS:1000521" => current_f32 = true,
                                b"MS:1000523" => current_f32 = false,
                                b"MS:1000574" => current_compressed = true,
                                b"MS:1000514" => current_array_kind = ArrayKind::Mz,
                                b"MS:1000515" => current_array_kind = ArrayKind::Intensity,
                                _ => {}
                            }
                        }
                    }
                    b"binary" if !self_closing => {
                        let text_start = gt + 1;
                        let Some(rel_end) = memmem::find(&slice[text_start..], b"</binary>") else { return None; };
                        let text_end = text_start + rel_end;
                        if current_array_kind != ArrayKind::Other {
                            let text = slice[text_start..text_end].trim_ascii();
                            if let Ok(raw) = BASE64.decode(text) {
                                let decompressed = if current_compressed {
                                    let mut out = vec![0u8; raw.len() * 5];
                                    unsafe { out.set_len(out.capacity()); }
                                    let mut decompressor = libdeflater::Decompressor::new();
                                    if let Ok(n) = decompressor.zlib_decompress(&raw, &mut out) {
                                        out.truncate(n);
                                        out
                                    } else {
                                        raw
                                    }
                                } else {
                                    raw
                                };

                                match current_array_kind {
                                    ArrayKind::Mz => {
                                        if current_f32 {
                                            for c in decompressed.chunks_exact(4) {
                                                mz_array.push(f32::from_le_bytes(c.try_into().unwrap()) as f64);
                                            }
                                        } else {
                                            for c in decompressed.chunks_exact(8) {
                                                mz_array.push(f64::from_le_bytes(c.try_into().unwrap()));
                                            }
                                        }
                                    }
                                    ArrayKind::Intensity => {
                                        if current_f32 {
                                            for c in decompressed.chunks_exact(4) {
                                                intensity_array.push(f32::from_le_bytes(c.try_into().unwrap()));
                                            }
                                        } else {
                                            for c in decompressed.chunks_exact(8) {
                                                intensity_array.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
                                            }
                                        }
                                    }
                                    ArrayKind::Other => {}
                                }
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

    if ms_level == 1 && !mz_array.is_empty() && mz_array.len() == intensity_array.len() {
        Some(ParsedSpectrumItem::Ms1 {
            scan_no: scan_number.unwrap_or(default_id),
            rt: rt_sec.unwrap_or(0.0),
            mzs: mz_array,
            ints: intensity_array,
        })
    } else if let Some(pmz) = precursor_mz {
        Some(ParsedSpectrumItem::Ms2(Ms2PrecursorInfo {
            scan_id: scan_number.unwrap_or(default_id),
            precursor_mz: pmz,
            rt: rt_sec.unwrap_or(0.0),
            charge,
        }))
    } else {
        None
    }
}

fn find_attr<'a>(attrs: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let n = attrs.len();
    let mut i = 0;
    while i < n {
        while i < n && attrs[i].is_ascii_whitespace() { i += 1; }
        let key_start = i;
        while i < n && !attrs[i].is_ascii_whitespace() && attrs[i] != b'=' { i += 1; }
        let key_end = i;
        if key_start == key_end { break; }
        while i < n && attrs[i].is_ascii_whitespace() { i += 1; }
        if i >= n || attrs[i] != b'=' { break; }
        i += 1;
        while i < n && attrs[i].is_ascii_whitespace() { i += 1; }
        if i >= n { break; }
        let quote = attrs[i];
        if quote != b'"' && quote != b'\'' { break; }
        i += 1;
        let val_start = i;
        while i < n && attrs[i] != quote { i += 1; }
        let val_end = i;
        if i < n { i += 1; }
        if &attrs[key_start..key_end] == key {
            return Some(&attrs[val_start..val_end]);
        }
    }
    None
}
