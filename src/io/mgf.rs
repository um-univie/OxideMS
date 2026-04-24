//! This module provides high-performance parsing and writing of Mascot Generic Format (MGF) files.
//!
//! MGF is a plain-text format commonly used in mass spectrometry to store peak lists.
//! This implementation utilizes memory-mapping and parallel processing for maximum throughput.

use crate::spectrum::{MassSpectralRecord, Peak};
use compact_str::CompactString;
use fast_float::parse;
use memchr::memmem;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

/// An estimate of the average size (in bytes) of a single MGF spectrum.
/// Used to pre-allocate memory when memory-mapping files.
pub const MGF_SPECTRA_SIZE: usize = 5_000;

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

/// A high-performance parser for MGF files.
///
/// Uses memory-mapping and optional parallel processing via Rayon to parse mass spectra
/// quickly and efficiently, minimizing unnecessary allocations.
pub struct MgfParser {
    /// The configuration settings used by this parser.
    pub config: MgfParserConfig,
    /// Tracks the current global spectrum ID across multiple parse calls.
    current_id: usize,
}

impl Default for MgfParser {
    /// Creates a new `MgfParser` with default configuration settings.
    fn default() -> Self {
        Self {
            config: MgfParserConfig::default(),

            current_id: 0,
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
        }
    }

    /// Parses a block directly from a string slice. No intermediate allocations.
    fn parse_block_slice(&self, raw_block: &[u8], index: u64) -> Option<MassSpectralRecord> {
        let mut peaks = Vec::with_capacity(100);
        let mut metadata = Vec::with_capacity(10);
        let mut charge = 0;

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
                    // Convert bytes to string just for the charge parsing
                    let val_str = unsafe { std::str::from_utf8_unchecked(val) };

                    charge = if val_str.ends_with('+') {
                        val_str[..val_str.len() - 1].parse::<i8>().unwrap()
                    } else if val_str.ends_with('-') {
                        -val_str[..val_str.len() - 1].parse::<i8>().unwrap()
                    } else {
                        val_str.parse::<i8>().unwrap()
                    };

                    if self.config.parse_metadata {
                        metadata.push((
                            CompactString::new("CHARGE"),
                            CompactString::new(charge.to_string()),
                        ));
                    }
                    continue;
                }

                if self.config.parse_metadata {
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
            let mut spec = MassSpectralRecord::new(peaks, charge, index as usize);
            if self.config.parse_metadata {
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
                self.parse_block_slice(raw_block, record_index as u64 + self.current_id as u64)
            })
            .collect();
        self.current_id += spectra.len();

        Ok(spectra)
    }
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

    #[test]
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
