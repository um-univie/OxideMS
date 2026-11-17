//! `oxidems`: inspect, convert and annotate mass spectrometry files.
//!
//! Spectral clustering has its own CLI, RedshiftMS.

use std::collections::BTreeMap;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use oxidems_lib::io::mgf::MgfWriter;
use oxidems_lib::io::{SPECTRAL_FILE_EXTENSIONS, is_spectral_file, read_spectra};
use oxidems_lib::spectrum::MassSpectralRecord;

#[derive(Parser)]
#[command(
    name = "oxidems",
    version,
    about = "Inspect, convert and annotate mass spectrometry files"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarize spectral files: spectra per MS level, charges, precursor
    /// m/z and retention time ranges.
    Info {
        /// MGF or mzML files, or folders of them.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Write the spectra of one MS level to MGF.
    Convert {
        /// Input file (MGF or mzML).
        input: PathBuf,
        /// Output MGF file.
        #[arg(short, long)]
        output: PathBuf,
        /// MS level to keep (0 = all levels).
        #[arg(long, default_value_t = 2)]
        ms_level: u8,
    },
    /// List molecular formulas matching a mass.
    #[cfg(feature = "sirius")]
    Formula {
        /// Ion m/z (or the neutral mass with --neutral).
        mass: f64,
        /// Adduct of the ion, e.g. "[M+H]+", "[M-H]-", "[M+Na]+".
        #[arg(long, default_value = "[M+H]+")]
        adduct: String,
        /// Treat the mass as the neutral monoisotopic mass.
        #[arg(long)]
        neutral: bool,
        /// Mass tolerance in ppm.
        #[arg(long, default_value_t = 5.0)]
        ppm: f64,
        /// Element set.
        #[arg(long, value_enum, default_value_t = Elements::ChnopsClBr)]
        elements: Elements,
        /// Number of formulas to print (closest mass first).
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
    /// Rank molecular formulas for each MS/MS spectrum by fragmentation-tree
    /// score and write the best candidates as CSV.
    #[cfg(feature = "sirius")]
    Annotate(AnnotateArgs),
}

#[cfg(feature = "sirius")]
#[derive(clap::Args)]
struct AnnotateArgs {
    /// Input file (MGF or mzML).
    input: PathBuf,
    /// Output CSV (default: standard output).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Precursor adduct, e.g. "[M+H]+", "[M+Na]+", "[M-H]-". Default:
    /// [M+H]+ or [M-H]- by the spectrum's polarity. A spectrum's own ADDUCT
    /// field takes precedence.
    #[arg(long)]
    adduct: Option<String>,
    /// Precursor and fragment mass tolerance in ppm.
    #[arg(long, default_value_t = 10.0)]
    ppm: f64,
    /// Candidate element set.
    #[arg(long, value_enum, default_value_t = Elements::ChnopsClBr)]
    elements: Elements,
    /// Most intense fragment peaks used per tree.
    #[arg(long, default_value_t = 60)]
    max_peaks: usize,
    /// Tree solver for the shortlisted candidates.
    #[arg(long, value_enum, default_value_t = Solver::Ilp)]
    solver: Solver,
    /// Candidates re-solved with --solver after a Critical-Path pass over all
    /// candidates (0 = solve every candidate with --solver).
    #[arg(long, default_value_t = 20)]
    exact_top_n: usize,
    /// Candidates written per spectrum.
    #[arg(long, default_value_t = 3)]
    top: usize,
    /// Only annotate the first N MS/MS spectra.
    #[arg(long)]
    limit: Option<usize>,
}

#[cfg(feature = "sirius")]
#[derive(Clone, Copy, clap::ValueEnum)]
enum Solver {
    /// Exact maximum colorful subtree (integer program).
    Ilp,
    /// Colour-subset dynamic programming; Critical-Path above 14 colours.
    Exact,
    /// Critical-Path heuristic only.
    CriticalPath,
}

#[cfg(feature = "sirius")]
#[derive(Clone, Copy, clap::ValueEnum)]
enum Elements {
    /// C, H, N, O
    Chno,
    /// C, H, N, O, P, S
    Chnops,
    /// C, H, N, O, P, S, Cl, Br (SIRIUS `-e SBrCl`)
    ChnopsClBr,
    /// C, H, N, O, P, S, F, Cl, Br
    ChnopsFClBr,
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Info { inputs } => info(&inputs),
        Command::Convert {
            input,
            output,
            ms_level,
        } => convert(&input, &output, ms_level),
        #[cfg(feature = "sirius")]
        Command::Formula {
            mass,
            adduct,
            neutral,
            ppm,
            elements,
            top,
        } => annotation::formula(mass, &adduct, neutral, ppm, elements, top),
        #[cfg(feature = "sirius")]
        Command::Annotate(args) => annotation::annotate(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The spectral files named by `inputs`; folders are listed non-recursively.
fn spectral_files(inputs: &[PathBuf]) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(input)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|path| path.is_file() && is_spectral_file(&path.to_string_lossy()))
                .collect();
            entries.sort();
            if entries.is_empty() {
                eprintln!(
                    "warning: no {} files in {}",
                    SPECTRAL_FILE_EXTENSIONS.join("/"),
                    input.display()
                );
            }
            files.extend(entries);
        } else {
            files.push(input.clone());
        }
    }
    Ok(files)
}

fn load(path: &Path) -> io::Result<Vec<MassSpectralRecord>> {
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} is not a file", path.display()),
        ));
    }
    read_spectra(&path.to_string_lossy(), true)
}

fn metadata_f64(record: &MassSpectralRecord, key: &str) -> Option<f64> {
    record
        .get_metadata(key)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn precursor_mz(record: &MassSpectralRecord) -> Option<f64> {
    metadata_f64(record, "PEPMASS").filter(|mz| *mz > 0.0)
}

#[derive(Default)]
struct Range(Option<(f64, f64)>);

impl Range {
    fn add(&mut self, value: f64) {
        self.0 = Some(match self.0 {
            Some((lo, hi)) => (lo.min(value), hi.max(value)),
            None => (value, value),
        });
    }
    fn show(&self, scale: f64, digits: usize) -> String {
        match self.0 {
            Some((lo, hi)) => format!("{:.digits$} - {:.digits$}", lo * scale, hi * scale),
            None => "-".to_string(),
        }
    }
}

fn info(inputs: &[PathBuf]) -> io::Result<()> {
    for path in spectral_files(inputs)? {
        let started = std::time::Instant::now();
        let records = load(&path)?;
        let seconds = started.elapsed().as_secs_f64();

        let mut levels: BTreeMap<u8, usize> = BTreeMap::new();
        let mut charges: BTreeMap<i8, usize> = BTreeMap::new();
        let (mut precursor, mut rt) = (Range::default(), Range::default());
        let mut peak_counts = Vec::new();
        let mut empty = 0usize;
        for record in &records {
            *levels.entry(record.ms_level).or_default() += 1;
            if let Some(time) = metadata_f64(record, "RTINSECONDS") {
                rt.add(time);
            }
            if record.ms_level == 1 {
                continue;
            }
            *charges.entry(record.get_charge()).or_default() += 1;
            if let Some(mz) = precursor_mz(record) {
                precursor.add(mz);
            }
            peak_counts.push(record.peaks().len());
            empty += usize::from(record.peaks().is_empty());
        }
        peak_counts.sort_unstable();

        println!("{}", path.display());
        println!(
            "  spectra          {} (read in {seconds:.2} s)",
            records.len()
        );
        for (level, count) in &levels {
            println!("  MS{level}              {count}");
        }
        let charges: Vec<String> = charges
            .iter()
            .map(|(z, n)| {
                if *z == 0 {
                    format!("unknown: {n}")
                } else {
                    format!("{z:+}: {n}")
                }
            })
            .collect();
        println!(
            "  MS/MS charges    {}",
            if charges.is_empty() {
                "-".into()
            } else {
                charges.join(", ")
            }
        );
        println!("  precursor m/z    {}", precursor.show(1.0, 4));
        println!("  retention time   {} min", rt.show(1.0 / 60.0, 2));
        if let Some(median) = peak_counts.get(peak_counts.len() / 2) {
            println!("  MS/MS peaks      median {median}, empty spectra {empty}");
        }
    }
    Ok(())
}

fn convert(input: &Path, output: &Path, ms_level: u8) -> io::Result<()> {
    let mut records: Vec<MassSpectralRecord> = load(input)?
        .into_iter()
        .filter(|record| ms_level == 0 || record.ms_level == ms_level)
        .collect();
    for record in &mut records {
        // mzML records keep the charge out of their metadata.
        let charge = record.get_charge();
        if charge != 0 && record.get_metadata("CHARGE").is_none() {
            let sign = if charge > 0 { '+' } else { '-' };
            record.insert_metadata("CHARGE", &format!("{}{sign}", charge.unsigned_abs()));
        }
    }
    MgfWriter::new().write_to_file(&output.to_string_lossy(), &records)?;
    eprintln!("wrote {} spectra to {}", records.len(), output.display());
    Ok(())
}

#[cfg(feature = "sirius")]
mod annotation {
    use super::*;
    use oxidems_lib::sirius::{
        Adduct, ChemicalRules, Formula, FormulaGenerator, FragmentationTreeBuilder,
        SiriusClusteringWorkflow, SiriusWorkflowConfig, SolverKind, TreeSolverConfig,
    };
    use oxidems_lib::workflow::MassTolerance;
    use rayon::prelude::*;

    fn parse_adduct(text: &str) -> io::Result<Adduct> {
        text.parse::<Adduct>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
    }

    /// Candidate formulas for `neutral_mass` from the chosen element set.
    fn candidates(elements: Elements, neutral_mass: f64, tolerance: MassTolerance) -> Vec<Formula> {
        let rules = ChemicalRules::default();
        match elements {
            Elements::Chno => {
                FormulaGenerator::chno().generate_candidates(neutral_mass, tolerance, &rules)
            }
            Elements::Chnops => {
                FormulaGenerator::chnops().generate_candidates(neutral_mass, tolerance, &rules)
            }
            Elements::ChnopsClBr => FormulaGenerator::chnops_halogens().generate_candidates(
                neutral_mass,
                tolerance,
                &rules,
            ),
            Elements::ChnopsFClBr => FormulaGenerator::chnops_fclbr().generate_candidates(
                neutral_mass,
                tolerance,
                &rules,
            ),
        }
    }

    pub fn formula(
        mass: f64,
        adduct: &str,
        neutral: bool,
        ppm: f64,
        elements: Elements,
        top: usize,
    ) -> io::Result<()> {
        let adduct = parse_adduct(adduct)?;
        let neutral_mass = if neutral {
            mass
        } else {
            adduct.ion_mz_to_neutral_mass(mass)
        };
        let mut formulas = candidates(elements, neutral_mass, MassTolerance::Ppm(ppm));
        let error_ppm = |f: &Formula| (f.exact_mass() - neutral_mass) / neutral_mass * 1e6;
        formulas.sort_by(|a, b| error_ppm(a).abs().total_cmp(&error_ppm(b).abs()));

        println!(
            "neutral mass {neutral_mass:.5}, {} formulas within {ppm} ppm",
            formulas.len()
        );
        println!(
            "{:<20} {:>12} {:>10} {:>6}",
            "formula", "mass", "ppm", "DBE"
        );
        for f in formulas.iter().take(top) {
            println!(
                "{:<20} {:>12.5} {:>10.2} {:>6.1}",
                f.to_string(),
                f.exact_mass(),
                error_ppm(f),
                f.dbe()
            );
        }
        Ok(())
    }

    /// The adduct to annotate `record` under: its own ADDUCT field, else
    /// `fallback`, else protonation or deprotonation by polarity and charge.
    fn adduct_for(record: &MassSpectralRecord, fallback: Option<&Adduct>) -> Option<Adduct> {
        if let Some(own) = record
            .get_metadata("ADDUCT")
            .and_then(|a| a.trim().parse::<Adduct>().ok())
        {
            return Some(own);
        }
        if let Some(adduct) = fallback {
            return Some(adduct.clone());
        }
        let charge = record.get_charge();
        let negative = charge < 0
            || record
                .get_metadata("IONMODE")
                .is_some_and(|m| m.trim().eq_ignore_ascii_case("negative"));
        let z = charge.unsigned_abs().max(1) as i8;
        Some(if negative {
            Adduct::deprotonated_with_charge(z)
        } else {
            Adduct::protonated_with_charge(z)
        })
    }

    pub fn annotate(args: &AnnotateArgs) -> io::Result<()> {
        let tolerance = MassTolerance::Ppm(args.ppm);
        let fallback = args.adduct.as_deref().map(parse_adduct).transpose()?;
        // The library's peak preprocessing (PeakPreprocessing::Simple: relative
        // intensity floor, precursor ceiling, top-N by intensity, dedupe).
        let preprocessor = SiriusClusteringWorkflow::new(SiriusWorkflowConfig {
            max_peaks_per_tree: args.max_peaks,
            tolerance,
            ..SiriusWorkflowConfig::default()
        });
        let solver = match args.solver {
            Solver::Ilp => SolverKind::Ilp,
            Solver::Exact => SolverKind::Exact,
            Solver::CriticalPath => SolverKind::CriticalPath,
        };
        let exact_top_n = (args.exact_top_n > 0).then_some(args.exact_top_n);

        let spectra: Vec<MassSpectralRecord> = load(&args.input)?
            .into_iter()
            .filter(|record| record.ms_level >= 2 && precursor_mz(record).is_some())
            .take(args.limit.unwrap_or(usize::MAX))
            .collect();
        eprintln!("annotating {} MS/MS spectra", spectra.len());

        let rows: Vec<Vec<String>> = spectra
            .par_iter()
            .map(|record| {
                let mz = precursor_mz(record).unwrap_or(0.0);
                let Some(adduct) = adduct_for(record, fallback.as_ref()) else {
                    return Vec::new();
                };
                let Some(peaks) =
                    preprocessor.preprocess_peaks_for_tree(record.peaks(), mz, record.get_charge())
                else {
                    return Vec::new();
                };
                let neutral_mass = adduct.ion_mz_to_neutral_mass(mz);
                let formulas = candidates(args.elements, neutral_mass, tolerance);
                let builder = FragmentationTreeBuilder::new(TreeSolverConfig {
                    tolerance,
                    adduct: adduct.clone(),
                    solver,
                    exact_top_n,
                    ..TreeSolverConfig::default()
                });
                let mut trees = builder.build_trees_for_candidates(&formulas, mz, &peaks);
                trees.sort_by(|a, b| b.1.total_score.total_cmp(&a.1.total_score));

                let title = record
                    .get_metadata("TITLE")
                    .or_else(|| record.get_metadata("NAME"))
                    .unwrap_or("")
                    .replace(['"', ','], " ");
                trees
                    .iter()
                    .take(args.top)
                    .enumerate()
                    .map(|(rank, (formula, tree))| {
                        let error_ppm = (formula.exact_mass() - neutral_mass) / neutral_mass * 1e6;
                        format!(
                            "{},{},{:.5},{},{},{},{},{:.2},{:.3},{:.3},{}",
                            record.id,
                            title,
                            mz,
                            record.get_charge(),
                            rank + 1,
                            formula,
                            adduct.name.replace(',', " "),
                            error_ppm,
                            tree.total_score,
                            tree.explained_intensity,
                            formulas.len(),
                        )
                    })
                    .collect()
            })
            .collect();

        let sink: Box<dyn Write> = match &args.output {
            Some(path) => Box::new(std::fs::File::create(path)?),
            None => Box::new(io::stdout().lock()),
        };
        let mut out = BufWriter::new(sink);
        writeln!(
            out,
            "index,title,precursor_mz,charge,rank,formula,adduct,mass_error_ppm,score,explained_intensity,candidates"
        )?;
        let annotated = rows.iter().filter(|r| !r.is_empty()).count();
        for line in rows.into_iter().flatten() {
            writeln!(out, "{line}")?;
        }
        out.flush()?;
        eprintln!(
            "{annotated} of {} spectra received a formula",
            spectra.len()
        );
        Ok(())
    }
}
