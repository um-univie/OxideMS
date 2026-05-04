//! Design-of-experiment grid search for metabolomics preprocessing.
//!
//! Usage:
//!   cargo run --example metabolomics_doe --release -- <folder> [purity_mz_tol] [purity_rt_tol_s]
//!
//! Metrics (per config):
//!   - weighted cluster purity  (higher = better separation between files/runs)
//!   - N10 cluster size         (higher = larger redundancy-capturing clusters)
//!   - spectra retained         (post-filter count)
//!   - cluster count

use itertools::Itertools;
use oxidems_lib::{
    clustering::cluster::{
        engine::{ClusterEngine, SpectrumMetadata},
        experiment::SpectralExperiment,
        flat_clusters::FlatClusters,
    },
    io::mgf::MgfParser,
    spectrum::MassSpectralRecord,
    workflow::{self, *},
};

// ── DoE axes ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Transform {
    None,
    Sqrt,
    Log,
}

impl Transform {
    fn name(self) -> &'static str {
        match self {
            Transform::None => "none",
            Transform::Sqrt => "sqrt",
            Transform::Log => "log",
        }
    }
}

#[derive(Clone, Debug)]
struct DoEConfig {
    transform: Transform,
    rel_intensity: f32,
    top_n: usize,
    min_peaks: usize,
    min_mz_range: f64,
    filter_mz_min: f64,
    filter_mz_max: f64,
    epsilon: f32,
    min_neighbors: usize,
    fragment_tol_da: f64,
}

impl DoEConfig {
    fn label(&self) -> String {
        format!(
            "transform={} rel_int={:.3} top_n={} min_peaks={} min_mz_range={} eps={:.2} min_nbrs={} frag_tol={:.3}",
            self.transform.name(),
            self.rel_intensity,
            self.top_n,
            self.min_peaks,
            self.min_mz_range,
            self.epsilon,
            self.min_neighbors,
            self.fragment_tol_da,
        )
    }

    fn build_workflow(&self) -> WorkFlowManager {
        let mut wf = WorkFlowManager::default();
        wf.use_parallel = false;
        wf.add_operation(FilterMZ::new(self.filter_mz_min, self.filter_mz_max));
        wf.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        wf.add_operation(FilterIntensityRelative::new(self.rel_intensity));
        match self.transform {
            Transform::None => {}
            Transform::Sqrt => wf.add_operation(SquareRootTransformIntensity::new()),
            Transform::Log => wf.add_operation(LogTransformIntensity::new()),
        }
        wf.add_operation(KeepTopNPeaks::new(self.top_n));
        wf.add_operation(MinNumberOfPeaks::new(self.min_peaks));
        wf.add_operation(MinMZRangeFilter::new(self.min_mz_range));
        wf
    }
}

// ── Raw file cache ────────────────────────────────────────────────────────────

struct RawFile {
    name: String,
    spectra: Vec<MassSpectralRecord>,
}

fn parse_folder(folder: &str) -> Vec<RawFile> {
    let mut paths: Vec<_> = std::fs::read_dir(folder)
        .expect("cannot read folder")
        .filter_map(|e| {
            let path = e.ok()?.path();
            if path.extension()?.to_str()? == "mgf" {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();

    paths
        .iter()
        .map(|path| {
            let mut parser = MgfParser::default();
            parser.config.use_parallel_parsing = false;
            let spectra = parser
                .parse_file(path.to_str().unwrap())
                .expect("parse failed");
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            RawFile { name, spectra }
        })
        .collect()
}

// ── Single config run ─────────────────────────────────────────────────────────

#[derive(Debug)]
struct RunResult {
    label: String,
    spectra_retained: usize,
    n_clusters: usize,
    n10: usize,
    purity: f64,
}

fn run_config(
    config: &DoEConfig,
    raw_files: &[RawFile],
    total_raw: usize,
    purity_mz_tol: f64,
    purity_rt_tol: f32,
) -> RunResult {
    let wf = config.build_workflow();
    let start_time = std::time::Instant::now();

    let mut all_spectra = Vec::new();
    let mut all_metadata: Vec<SpectrumMetadata> = Vec::new();
    let mut file_names: Vec<String> = Vec::new();

    // println!("Workflow: {:?}", wf);
    for (file_idx, raw) in raw_files.iter().enumerate() {
        file_names.push(raw.name.clone());
        let mut spectra = raw.spectra.clone();
        wf.execute(&mut spectra);

        for s in spectra
            .into_iter()
            .filter(|s| !s.peaks().is_empty() && s.get_charge() != 0)
        {
            let scan_number = s
                .get_metadata("SCANS")
                .unwrap_or("0")
                .parse::<usize>()
                .unwrap_or(0);
            let msn = s.to_msn();
            all_metadata.push(SpectrumMetadata {
                precursor_mz: msn.precursor_mz,
                retention_time_in_seconds: msn.retention_time_in_seconds,
                charge: msn.charge(),
                scan_number,
                file_index: file_idx,
            });
            all_spectra.push(msn);
        }
    }

    println!("Kept {} spectra", all_spectra.len());

    // assign global IDs
    for (i, s) in all_spectra.iter_mut().enumerate() {
        s.id = i as u64;
    }

    let spectra_retained = all_spectra.len();

    let mut engine = ClusterEngine::default();
    engine.set_mz_range(config.filter_mz_min, config.filter_mz_max);
    engine.hasher.tolerance = MassTolerance::Da(config.fragment_tol_da);
    engine.hasher.recalculate_lookup_table();
    engine.epsilon = config.epsilon;
    engine.min_neighbors = config.min_neighbors;
    engine.exact = true;

    let mut exp = SpectralExperiment {
        file_names,
        metadata: all_metadata,
        workflow: WorkFlowManager::default(),
        parser: MgfParser::default(),
        engine,
        spectra: all_spectra,
        filter_unknown_charge: false,
        total_spectra_ingested: total_raw,
    };

    let clusters = exp.cluster_spectra();
    let n_clusters = clusters.iter().count();
    let n10 = exp.calculate_n10_value(&clusters);
    let purity = if n_clusters > 0 {
        exp.average_weighted_cluster_purity(&clusters, purity_mz_tol, purity_rt_tol)
    } else {
        0.0
    };

    RunResult {
        label: config.label(),
        spectra_retained,
        n_clusters,
        n10,
        purity,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: metabolomics_doe <folder> [purity_mz_tol_da] [purity_rt_tol_s]");
        std::process::exit(1);
    }
    let folder = &args[1];
    let purity_mz_tol: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let purity_rt_tol: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30.0);

    eprintln!("Parsing MGF files from {}...", folder);
    let raw_files = parse_folder(folder);
    let total_raw: usize = raw_files.iter().map(|f| f.spectra.len()).sum();
    eprintln!(
        "  {} files, {} total spectra (raw)",
        raw_files.len(),
        total_raw
    );

    // ── DoE grid ─────────────────────────────────────────────────────────────
    let transforms = [Transform::None, Transform::Sqrt, Transform::Log];
    let rel_intensities: [f32; 3] = [0.001, 0.01, 0.05];
    let top_ns: [usize; 3] = [30, 50, 100];
    let min_peaks_vals: [usize; 2] = [3, 5];
    let min_mz_ranges: [f64; 2] = [50.0, 100.0];
    let epsilons: [f32; 5] = [0.05, 0.1, 0.2, 0.3, 0.4];
    let min_neighbors_vals: [usize; 2] = [2, 3];
    let fragment_tols: [f64; 3] = [0.02, 0.05, 0.1];

    let configs: Vec<DoEConfig> = transforms
        .iter()
        .cartesian_product(rel_intensities.iter())
        .cartesian_product(top_ns.iter())
        .cartesian_product(min_peaks_vals.iter())
        .cartesian_product(min_mz_ranges.iter())
        .cartesian_product(epsilons.iter())
        .cartesian_product(min_neighbors_vals.iter())
        .cartesian_product(fragment_tols.iter())
        .map(|(((((((t, ri), tn), mp), mr), eps), mn), ft)| DoEConfig {
            transform: *t,
            rel_intensity: *ri,
            top_n: *tn,
            min_peaks: *mp,
            min_mz_range: *mr,
            filter_mz_min: 50.0,
            filter_mz_max: 1500.0,
            epsilon: *eps,
            min_neighbors: *mn,
            fragment_tol_da: *ft,
        })
        .collect();

    let total = configs.len();
    eprintln!("Running {} configs...\n", total);

    // print CSV header
    println!(
        "transform,rel_intensity,top_n,min_peaks,min_mz_range,\
         epsilon,min_neighbors,fragment_tol_da,\
         spectra_retained,n_clusters,n10,purity"
    );

    let mut results: Vec<RunResult> = Vec::with_capacity(total);

    for (i, config) in configs.iter().enumerate() {
        eprint!("[{:>3}/{}] {} ... ", i + 1, total, config.label());
        let result = run_config(config, &raw_files, total_raw, purity_mz_tol, purity_rt_tol);
        eprintln!(
            "purity={:.4} n10={} clusters={}",
            result.purity, result.n10, result.n_clusters
        );

        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.6}",
            config.transform.name(),
            config.rel_intensity,
            config.top_n,
            config.min_peaks,
            config.min_mz_range,
            config.epsilon,
            config.min_neighbors,
            config.fragment_tol_da,
            result.spectra_retained,
            result.n_clusters,
            result.n10,
            result.purity,
        );

        results.push(result);
    }

    // ── Summary: top 10 by purity ─────────────────────────────────────────────
    results.sort_by(|a, b| b.purity.partial_cmp(&a.purity).unwrap());
    eprintln!("\n── Top 10 by weighted purity ────────────────────────────────");
    eprintln!(
        "{:<70} {:>8} {:>6} {:>6}",
        "config", "purity", "n10", "clusters"
    );
    for r in results.iter().take(10) {
        eprintln!(
            "{:<70} {:>8.4} {:>6} {:>6}",
            r.label, r.purity, r.n10, r.n_clusters
        );
    }
}
