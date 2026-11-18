<p align="center">
  <img src="https://raw.githubusercontent.com/um-univie/OxideMS/main/logo.png" alt="OxideMS" width="320"/>
</p>

<p align="center">High-performance mass spectrometry data processing and clustering in Rust.</p>

---

OxideMS is a Rust library for processing and clustering tandem mass spectra (MS/MS). It provides a composable workflow pipeline for spectrum preprocessing, a DBSCAN-based clustering engine with both exact and approximate similarity modes, diagnostic plotting, and a growing machine learning layer for spectral annotation.

The library targets large-scale experiments — hundreds of MGF files, millions of spectra — with parallel parsing, Rayon-backed processing, and optional GPU-accelerated similarity computation.

## Features

- **Parsing** — fast, memory-mapped MGF and mzML parsers with parallel ingestion across files
- **Workflow** — composable preprocessing pipeline with 16 built-in operations and parallelism via Rayon
- **Clustering** — DBSCAN over cosine similarity with exact (sparse matrix) and approximate (hashed) modes
- **Plotting** — single/mirror spectrum plots and a full batch of experiment-level diagnostic charts
- **Machine learning** — CNN-based isotope detector built on the [Burn](https://burn.dev) framework *(experimental)*

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxidems_lib = { git = "https://github.com/um-univie/OxideMS", branch = "main" }

# Optional features
# oxidems_lib = { git = "...", features = ["gpu"] }   # GPU-accelerated similarity
# oxidems_lib = { git = "...", features = ["cli"] }   # the `oxidems` command-line tool
```

The default `sirius` feature builds the SIRIUS-style annotation stack
(`oxidems_lib::sirius`), which pulls in candle, good_lp and the
[oxidane](https://github.com/um-univie/oxidane) cheminformatics crate (fetched
from GitHub at a pinned revision). Clustering-only users should turn it off:

```toml
oxidems_lib = { git = "https://github.com/um-univie/OxideMS", branch = "main", default-features = false }
```

### Developing with a local oxidane

To work on oxidane and OxideMS together, override the pinned revision with a
checkout next to this repository. Put this in `.cargo/config.toml` of your own
workspace, or pass it per command, and do not commit it:

```bash
cargo build --config 'patch."https://github.com/um-univie/oxidane".oxidane.path="../oxidane"'
```

### Tests

```bash
cargo test --release
```

runs on a fresh clone. `test_files/test1.mgf` holds two MSnLib spectra (see
`test_files/README.md`). Three tests need a 600,566-spectrum MGF that is too
large for git; they are marked `#[ignore]`. Place it at `test_files/test2.mgf`
and run `cargo test --release -- --ignored`.

### Examples

`examples/` holds benchmarks and research tools. Each one prints its usage and
the data or model it needs when run without arguments, for example
`cargo run --release --example dataset_overview`. Most need public datasets
(GNPS, MoNA, MSnLib, MassIVE/PRIDE runs) or a fingerprint model trained with
`train_fingerprint_model`; none of these are in the repository.

## Command-line tool

The `cli` feature builds an `oxidems` binary for quick work on spectral files.
Clustering has its own CLI, [RedshiftMS](https://github.com/um-univie/RedshiftMS).

```bash
cargo install --git https://github.com/um-univie/OxideMS --features cli

oxidems info data/                        # spectra per MS level, charges, m/z and RT ranges
oxidems convert run.mzML -o run.mgf       # MS/MS spectra to MGF (--ms-level 0 keeps all)
oxidems formula 195.0877                  # formulas for an [M+H]+ ion within 5 ppm
oxidems annotate library.mgf -o out.csv   # top formulas per MS/MS spectrum by tree score
```

`annotate` scores every candidate formula with a fragmentation tree and, by
default, re-solves the 20 best exactly (`--solver ilp --exact-top-n 20`). It
uses the precursor m/z as measured. Run `oxidems <command> --help` for the
options.

---

## Parsing

MGF (Mascot Generic Format) and mzML are both supported input formats, dispatched automatically by file extension (`oxidems_lib::io::read_spectra`) — a single input folder may freely mix `.mgf` and `.mzML` files. Both parsers use memory-mapped I/O with no intermediate string allocations, making them suitable for very large files.

```rust
use oxidems_lib::io::mgf::MgfParser;

let mut parser = MgfParser::default();
let spectra = parser.parse_file("experiment.mgf").unwrap();
println!("Parsed {} spectra", spectra.len());
```

For a folder of MGF files processed in parallel across files (each file parsed sequentially to avoid contention):

```rust
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;

let mut exp = SpectralExperiment::proteomics();
exp.ingest_files_parallel("data/mgf_files/");
```

MGF metadata fields (e.g. `PEPMASS`, `CHARGE`, `RTINSECONDS`, `SCANS`) are parsed into a per-spectrum key-value store and accessed with `spectrum.get_metadata("PEPMASS")`.

Writing processed spectra back to MGF:

```rust
use oxidems_lib::io::mgf::MgfWriter;

let writer = MgfWriter::default();
writer.write_to_file(&spectra, "processed.mgf").unwrap();
```

---

## Workflow

The workflow system is a sequential pipeline of operations applied to a slice of `MassSpectralRecord`s. Each step implements either `Operation` (sees all spectra) or the more common `SingleOperation` (one spectrum at a time), which gets parallel dispatch via Rayon for free.

```rust
use oxidems_lib::MassSpectralRecord;
use oxidems_lib::workflow::*;

let mut wfm = WorkFlowManager::new();
wfm.add_operation(FilterMZ::new(101.0, 1500.0));
wfm.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
wfm.add_operation(FilterIntensityRelative::new(0.01));
wfm.add_operation(SquareRootTransformIntensity::new());
wfm.add_operation(KeepTopNPeaks::new(50));
wfm.add_operation(MinNumberOfPeaks::new(5));
wfm.add_operation(MinMZRangeFilter::new(250.0));

wfm.execute(&mut spectra);
```

### Available operations

| Operation | Description |
|---|---|
| `FilterMZ` | Remove peaks outside an m/z range |
| `FilterIntensity` | Remove peaks outside an absolute intensity range |
| `FilterIntensityRelative` | Remove peaks below a fraction of the base peak |
| `KeepTopNPeaks` | Keep the N most intense peaks, re-sorted by m/z |
| `Normalize` | Scale all intensities so the base peak equals 1.0 |
| `SquareRootTransformIntensity` | Compress dynamic range with a √ transform |
| `CentroidPeaks` | Merge adjacent peaks within a m/z tolerance into intensity-weighted centroids |
| `Deisotope` | Collapse C13 isotope satellites into the monoisotopic peak |
| `RemovePrecursor` | Remove the precursor ion and its charge-state isotopes |
| `DropUnknownCharge` | Discard spectra whose precursor charge is unknown (0) |
| `ReplacePrecursor` | Set `PEPMASS` to the highest m/z peak in the spectrum |
| `MinNumberOfPeaks` | Discard spectra with fewer than N peaks |
| `MinMZRangeFilter` | Discard spectra whose m/z span is below a threshold |
| `AbsoluteIntensityFilter` | Discard spectra whose base peak is below a threshold |
| `LocalSNRFilter` | Remove peaks below SNR × estimated noise floor (25th percentile) |
| `PeakDensityFilter` | Discard spectra with more than N peaks per 100 Da window |
| `SortPeaks` | Sort peaks by m/z (applied automatically by standard constructors) |

`MassTolerance` can be expressed as `Da(f64)` or `Ppm(f64)` and is used by `RemovePrecursor`, `Deisotope`, and the clustering engine.

### Default experiment workflows

Three ready-made preprocessing profiles are provided via `SpectralExperiment`:

**`SpectralExperiment::default()`**
A general-purpose baseline for MS/MS data. Suitable as a starting point when the experiment type is unknown. These defaults were adopted from the falcon project and are subject to change.
```
FilterMZ(101–1500 Da) → RemovePrecursor(±1.5 Da) → FilterIntensityRelative(1%)
→ KeepTopNPeaks(50) → MinNumberOfPeaks(5) → MinMZRangeFilter(250 Da)
```

**`SpectralExperiment::proteomics()`**
Raises the minimum m/z to 200 Da to exclude low-mass immonium-ion noise common in peptide fragmentation, and removes common laboratory contaminants. Tuned by a DOE study against three real proteomics datasets with MS-GF+ ground truth; `epsilon = 0.05`.
```
FilterMZ(200–1500 Da) → RemovePrecursor(±1.5 Da) → RemoveCommonContaminants(20 ppm)
→ FilterIntensityRelative(1%) → KeepTopNPeaks(50) → MinNumberOfPeaks(5) → MinMZRangeFilter(250 Da)
```

**`SpectralExperiment::metabolomics()`**
Lowers the minimum m/z to 50 Da to retain small metabolite fragment ions and relaxes the peak-count/m-z-range floors. Tuned by a DOE study against real MS-RT-proxy-scored acquisition data; no intensity transform (an earlier square-root transform was dropped after re-validation against real data showed it hurt completeness); `epsilon = 0.5`.
```
FilterMZ(50–1500 Da) → RemovePrecursor(±1.5 Da)
→ KeepTopNPeaks(50) → MinNumberOfPeaks(3) → MinMZRangeFilter(50 Da)
```

Building a custom profile on top of an existing one is straightforward e.g.:

```rust
let mut exp = SpectralExperiment::proteomics();
exp.workflow.add_operation(Deisotope::new(3, 0.02, 5));
```

---

## Clustering

Clustering groups spectra that share a similar fragmentation pattern — useful for redundancy reduction, chimeric spectrum detection, and spectral library building.

### Basic usage

```rust
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;

let mut exp = SpectralExperiment::proteomics();

// Ingest all MGF files in a folder (files processed in parallel)
exp.ingest_files_parallel("data/mgf_files/");
// Alternatively one can use file level parallelism to reduce memory requirements during parsing
exp.ingest_files("data/mgf_files/");
// Parallelism can be completely disabled by setting the corresponding flag
exp.parser.use_parallel_parsing(false);
exp.ingest_files("data/mgf_files/");

// Partition spectra by charge and precursor m/z, then run DBSCAN and refine the clusters afterwards
let clusters = exp.cluster_spectra();

println!(
    "Found {} clusters from {} spectra",
    clusters.active_cluster_count(),
    exp.total_spectra_ingested
);

// Write results to CSV (mzspec USI format)
exp.write_results(&clusters, "results.csv");
```

The output CSV contains one row per clustered spectrum with columns: `identifier`, `index`, `title`, `precursor_charge`, `precursor_mz`, `retention_time`, `cluster`.

- `identifier` is `mzspec:USI000000:<file>:scan:<SCANS>`, or `...:index:<index>` for a spectrum without a parseable scan number. Scan numbers can repeat within a file; `index` (the 0-based position of the spectrum in its file, counted before filtering) cannot.
- `title` is the spectrum's `TITLE`, quoted if it contains a comma.
- With `exp.engine.representative_floor` set, each cluster lists its representative (the medoid) first and two more columns follow: `representative` (`true` for it) and `rep_cosine` (each member's exact cosine to it). Members below the floor are split off; a spectrum left alone is unclustered.
- Set `exp.emit_unclustered = true` before ingesting to also write singletons and spectra removed by preprocessing, with `cluster` = -1, so every input spectrum gets a row.

### How clustering works

1. **Partitioning** — `prepare_clustering()` splits spectra into sub-groups by charge state and precursor m/z window (and optionally retention time). Each partition is clustered independently. A charge is only assigned when the file states one or the precursor isotope spacing shows it; spectra left at charge 0 (unknown) form their own groups, or are discarded with `DropUnknownCharge`.

2. **Similarity** — cosine similarity between hashed or binned intensity vectors.
   - *Approximate mode* (default): spectra are projected into a 4096-dimensional hash space using xxHash32. Similarity is a fast dot product — O(1) per pair after hashing.
   - *Exact mode* (`engine.exact = true`): sparse matrix multiplication over m/z-binned spectra via `faer`. More accurate, higher memory use.

3. **DBSCAN** — standard density-based clustering with configurable `epsilon` (cosine distance threshold) and `min_neighbors`.
   With `min_neighbors = 2` this links a whole chain of spectra through pairs above the threshold (single linkage). Set `engine.complete_linkage` to split each cluster by complete-linkage agglomerative clustering on the exact cosine, so every pair in a cluster reaches the threshold.

4. **Post-hoc refinement** — after DBSCAN, clusters whose precursor m/z span exceeds the engine tolerance are split using agglomerative hierarchical clustering, reducing false positives.

### Engine configuration

```rust
use oxidems_lib::clustering::cluster::experiment::SpectralExperiment;
use oxidems_lib::workflow::MassTolerance;

let mut exp = SpectralExperiment::proteomics();
exp.engine.epsilon = 0.05;                                // cosine distance threshold
exp.engine.min_neighbors = 2;                             // DBSCAN minimum neighbours
exp.engine.min_shared_peaks = 4;                          // pairs must also share >= 4 fragment peaks (0 = off)
exp.engine.representative_floor = Some(0.9);              // medoid per cluster; split off members below cosine 0.9
exp.engine.complete_linkage = true;                       // every pair in a cluster must reach 1 - epsilon
exp.engine.mz_tolerance = MassTolerance::Da(0.02);        // precursor m/z window for partitioning
exp.engine.exact = false;                                 // use approximate (hashed) mode
exp.engine.set_mz_range(40.0, 2000.0);                    // hashed fragment range; match the workflow's FilterMZ
exp.engine.set_fragment_tolerance(MassTolerance::Da(0.05)); // hashed fragment bin width
```

Change the hasher through `set_mz_range`, `set_fragment_tolerance` and
`set_hash_size`: assigning `engine.hasher` fields directly leaves its bin
lookup table stale. Fragments outside the hasher's m/z range are ignored.

### Quality metrics

```rust
// Weighted cluster purity (fraction of spectra from the same precursor in each cluster)
let purity = exp.average_weighted_cluster_purity(&clusters, 0.02, 30.0);
println!("Weighted purity: {:.3}", purity);

// N10 value: size of the smallest cluster containing the top 10% of clustered spectra
let n10 = exp.calculate_n10_value(&clusters);
println!("N10: {}", n10);
```

---

## Plotting

### Single and mirror spectrum plots

```rust
use oxidems_lib::plotting::SpectrumPlotter;

let plotter = SpectrumPlotter::new()
    .dimensions(1200, 600)
    .title("Example Spectrum")
    .show_statistics(true);

// Single spectrum
plotter.plot_single(&spectrum, "spectrum.png").unwrap();

// Mirror plot — matching peaks highlighted in red
plotter.plot_mirror(&query, &reference, "mirror.png").unwrap();
```

### Experiment-level diagnostics

`SpectrumAnalyzer` produces a battery of diagnostic plots from a collection of `MSNSpectrum` objects.

```rust
use oxidems_lib::plotting::SpectrumAnalyzer;

let analyzer = SpectrumAnalyzer::new(spectra);
analyzer.plot_mz_rt("mz_rt.png").unwrap();
analyzer.plot_tic("tic.png").unwrap();
analyzer.plot_charge_distribution("charges.png").unwrap();
analyzer.plot_intensity_distribution("intensities.png").unwrap();
analyzer.plot_kendrick_mass_defect("kmd.png").unwrap();
```

Available plots:

| Method | Description |
|---|---|
| `plot_mz_rt` | Precursor m/z vs. retention time scatter |
| `plot_mz_kde` | 2D KDE heatmap of m/z × RT density |
| `plot_tic` | Total ion current over retention time |
| `plot_peak_counts` | Histogram of per-spectrum peak counts |
| `plot_feature_map` | 2D intensity heatmap (m/z × RT) |
| `plot_feature_map_log` | Log-scale version of the feature map |
| `plot_intensity_distribution` | Histogram of all peak intensities |
| `plot_charge_distribution` | Bar chart of precursor charge states |
| `plot_neutral_mass_rt` | Neutral mass vs. retention time |
| `plot_mz_rt_by_charge` | m/z vs. RT faceted by charge state |
| `plot_3d_landscape` | 3D surface of m/z × RT intensity |
| `plot_kendrick_mass_defect` | Kendrick mass defect — CHO composition diagnostic |

---

## Machine learning *(experimental)*

The `ml` feature flag enables a CNN-based model for isotope peak detection, built on the [Burn](https://burn.dev) deep learning framework with a WGPU backend for GPU training.

```toml
oxidems_lib = { path = "...", features = ["ml"] }
```

The current model (`IsotopeDetector`) is a lightweight 1D convolutional network that classifies peaks in 40 Da windows as noise, monoisotopic, or isotope. Training uses labeled windows derived from known spectra.

Planned ML capabilities include:

- Spectral library matching with learned similarity embeddings
- De novo peptide/metabolite sequencing support
- Charge state and adduct prediction

This module is under active development and the API is not yet stable.

---

## Performance notes

- Parsing scales linearly: `ingest_files_parallel` distributes files across threads while each file is parsed sequentially to avoid I/O bottlenecks.
- The approximate clustering mode is the default. Exact mode (`engine.exact = true`) offers higher recall at the cost of more compute for large datasets, by default approximation is still used for very large partitions (>30000 spectra per partition, which usually happens around 10 million spectra total), to change this simply increase the approximate threshold to usize::MAX which effectively disables approximation.
- GPU similarity computation is available via the `gpu` feature flag (WGPU backend), providing large speedups for batches of tens of thousands of spectra.
- Release builds strip debug symbols and use `opt-level = 3`; enabling `codegen-units = 1` in `Cargo.toml` can yield additional gains for sustained throughput workloads.

---

## License

OxideMS is MIT-licensed (see [LICENSE](LICENSE)), except four files in
`src/sirius/` that are ported from SIRIUS and are licensed under the
LGPL-3.0-or-later. They are only compiled with the `sirius` feature (on by
default); RedshiftMS and other clustering-only users who disable it build
nothing but MIT code. [ATTRIBUTION.md](ATTRIBUTION.md) lists them along with all
other third-party attributions.
