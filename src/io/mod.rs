pub mod mgf;
pub mod mzml;

use std::io;

use crate::spectrum::MassSpectralRecord;

/// Spectral file extensions [`read_spectra`] understands (compared
/// case-insensitively).
pub const SPECTRAL_FILE_EXTENSIONS: &[&str] = &["mgf", "mzml"];

/// Whether `path` ends with one of [`SPECTRAL_FILE_EXTENSIONS`].
pub fn is_spectral_file(path: &str) -> bool {
    path.rsplit_once('.').is_some_and(|(_, ext)| {
        SPECTRAL_FILE_EXTENSIONS.iter().any(|known| ext.eq_ignore_ascii_case(known))
    })
}

/// Parses a spectral file with the parser matching its extension: `.mzML`
/// or MGF for anything else. `parallel` controls each parser's own per-file
/// parallelism; pass `false` when the caller already parallelizes across
/// files.
pub fn read_spectra(path: &str, parallel: bool) -> io::Result<Vec<MassSpectralRecord>> {
    let extension = path.rsplit_once('.').map(|(_, ext)| ext.to_ascii_lowercase());
    match extension.as_deref() {
        Some("mzml") => {
            let mut config = mzml::MzmlParserConfig::default();
            config.use_parallel_parsing = parallel;
            mzml::MzmlParser::new(config).parse_file(path)
        }
        _ => {
            let mut parser = mgf::MgfParser::default();
            parser.config.use_parallel_parsing = parallel;
            parser.parse_file(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_extensions_match_case_insensitively() {
        assert!(is_spectral_file("a/run.MGF"));
        assert!(is_spectral_file("run.mzML"));
        assert!(!is_spectral_file("run.csv"));
        assert!(!is_spectral_file("mgf"));
    }

    #[test]
    fn other_extensions_are_read_as_mgf() {
        assert!(!read_spectra("test_files/test2.mgf", false).unwrap().is_empty());
    }
}
