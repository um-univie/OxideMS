# Third-party attributions

OxideMS is MIT-licensed (see `LICENSE`), with the exceptions listed under
"Code ported from SIRIUS" below.

## Code ported from SIRIUS (LGPL-3.0-or-later)

These files are ported to Rust from the SIRIUS source code. They are therefore
distributed under the GNU Lesser General Public License, version 3 or (at your
option) any later version, and not under the MIT license:

| File | What it contains |
|---|---|
| `src/sirius/sirius_scoring.rs` | the fragmentation-tree scoring objective, its loss and fragment scorers, and the MS1 isotope-pattern scoring |
| `src/sirius/sirius_scoring_tables.rs` | the scoring parameter tables of the SIRIUS default profile |
| `src/sirius/sirius_svm.rs` | the trained linear formula prior |
| `src/sirius/solver.rs` | the tree-size auto-tuning (Beautification) and recalibration defaults, next to this project's own solvers |

The license texts are in `LICENSES/LGPL-3.0-or-later.txt` and
`LICENSES/GPL-3.0-or-later.txt` (the LGPL grants additional permissions on top
of the GPL, so both apply). Each file starts with a notice naming the original
copyright holders and stating that it was modified. These files are only
compiled with the `sirius` Cargo feature (on by default); a build with
`default-features = false` contains no LGPL code.

SIRIUS is developed by the Chair of Bioinformatics, Friedrich-Schiller
University Jena. The source files these ports come from carry the notice
"Copyright (C) 2013-2020 Kai Dührkop, Markus Fleischauer, Marcus Ludwig, Martin
A. Hoffman, Fleming Kretschmer and Sebastian Böcker" and are licensed under the
LGPL-3.0-or-later. The SIRIUS application as a whole is licensed under the
AGPL-3.0; nothing here is taken from parts of it that are only available under
the AGPL. Source: <https://github.com/sirius-ms/sirius>.

OxideMS is not affiliated with or endorsed by the SIRIUS developers.

### What was ported, and from where

**Fragmentation-tree scoring** (`sirius_scoring.rs`, selected via
`TreeSolverConfig::scoring = ScoringModel::Sirius`, the default). The loss and
fragment scorers of the default profile, `CommonRootLossScorer` and, at the
root, the formula prior below. The parameter tables in `sirius_scoring_tables.rs`
(loss-, fragment- and root-loss score dictionaries, the log-normal loss-size
fit, the radical and strange-element whitelists, the implausible-loss list) are
transcribed by `scratch/gen_sirius_scoring.py` from
`fragmentation_tree/fragmentation_tree_construction/src/main/resources/profiles/default.json`.
`COMMON_LOSS_RECOMBINED` is the output of SIRIUS's `MinimalScoreRecombinator`
(`CommonLossEdgeScorer.java`) applied to the common-loss dictionary.

**Formula prior** (`sirius_svm.rs`). `SupportVectorMolecularFormulaScorer.java`
(`chemistry_base`): the two normalisation vectors, the weight vector, the
~550-entry whitelist and the decision rule, with the 60 features of
`FormulaFeatureVector.java`.

**MS1 isotope-pattern scoring** (`sirius_scoring.rs::isotope_score`).
`IsotopePatternAnalysis` (`extractPattern`, `scoreFormulas`),
`PeriodicTable.getIsotopicMassWindow` and the four default pattern scorers
(`MassDifferenceDeviationScorer`, `MassDeviationScorer`,
`NormalDistributedIntensityScorer`, `MissingPeakScorer`), with the default
configuration's parameters (`MS1MassDeviation` 10/10/5 ppm, σ_abs 0.02 /
σ_rel 0.08, λ 50, cutoff 0.01).

**Tree-size auto-tuning and recalibration** (`solver.rs`). `Beautification`
from `FasterTreeComputationInstance` (`estimateTreeSizeAndRecalibration`,
`isHighQuality`) with its constants (`MAX_TREESIZE` 2.5, `TREE_SIZE_INCREASE`
1.0, `MIN_EXPLAINED_INTENSITY` 0.7, `MIN_NUMBER_OF_EXPLAINED_PEAKS` 15), and the
`HypothesenDrivenRecalibration` defaults. Some of these values were read from
the compiled SIRIUS 5.6.3 release of these LGPL classes.

**Not ported**: `MultimereLossScorer` and adduct-switch losses. Under
`ScoringModel::Native` a fixed `TreeSolverConfig::tree_size_bonus` replaces
Beautification.

The model itself is described in Böcker, S. & Dührkop, K. *Fragmentation trees
reloaded.* J. Cheminform. **8**, 5 (2016),
<https://doi.org/10.1186/s13321-016-0116-8>, and the earlier work it builds on
(Böcker & Rasche, *Bioinformatics* 24, i49 (2008); Rasche et al., *Anal. Chem.*
83, 1243 (2011); Dührkop et al., *WABI* 2018 for the Critical-Path heuristic).

## ClassyFire chemical taxonomy

`src/sirius/canopus/ontology.rs` contains a subset of ChemOnt, the ClassyFire
chemical taxonomy (identifiers, names and parent relations of 81 classes):

> Djoumbou Feunang, Y. et al. *ClassyFire: automated chemical classification
> with a comprehensive, computable taxonomy.* J. Cheminform. **8**, 61 (2016).
> <https://doi.org/10.1186/s13321-016-0174-y>

## Tuffy font

`src/assets/Tuffy.ttf`, embedded by `src/plotting.rs` for plot labels, is the
Tuffy typeface by Thatcher Ulrich, released into the public domain.

## Everything else

The rest of `src/sirius/` (formula decomposition, the Native-model scorers, the
feature extractor, the fingerprint predictor and search, the compound-class
classifier) is this project's own implementation of methods from the SIRIUS,
CSI:FingerID and CANOPUS literature, cited inline in the source, and is
MIT-licensed. `scratch/gen_sirius_scoring.py` is MIT-licensed as well; the file
it generates is LGPL-3.0-or-later.
