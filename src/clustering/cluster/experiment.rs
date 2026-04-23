use super::dbscan::ClusterLabel;
use super::engine::{ClusterEngine, SpectrumMetadata};
use super::flat_clusters::FlatClusters;
use crate::{
    MSNSpectrum,
    io::mgf::{MGF_SPECTRA_SIZE, MgfParser, MgfParserConfig},
    workflow::*,
};
use nohash_hasher::IntMap;
use rayon::prelude::*;
use std::io::Write;

#[derive(Default)]
pub struct SpectralExperiment {
    pub file_names: Vec<String>,
    pub metadata: Vec<SpectrumMetadata>,
    pub workflow: WorkFlowManager,
    pub parser: MgfParser,
    pub engine: ClusterEngine,
    pub spectra: Vec<MSNSpectrum>,
    pub filter_unknown_charge: bool,
    pub total_spectra_ingested: usize,
}

impl SpectralExperiment {
    pub fn default() -> Self {
        let mut workflow = WorkFlowManager::default();
        workflow.add_operation(FilterMZ::new(101.0, 1500.0));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        Self {
            workflow,
            filter_unknown_charge: true,
            ..Default::default()
        }
    }

    pub fn proteomics() -> Self {
        let mut workflow = WorkFlowManager::default();
        let min_mz = 200.0;
        workflow.add_operation(FilterMZ::new(min_mz, 1500.0));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(SquareRootTransformIntensity::new());
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        let mut engine = ClusterEngine::default();
        engine.hasher.min_mz = min_mz;

        Self {
            workflow,
            engine: ClusterEngine::default(),
            parser: MgfParser::default(),
            filter_unknown_charge: true,
            ..Default::default()
        }
    }

    pub fn metabolomics() -> Self {
        let mut workflow = WorkFlowManager::default();
        workflow.add_operation(FilterMZ::new(50.0, 1500.0));
        workflow.add_operation(RemovePrecursor::new(MassTolerance::Da(1.5), 0));
        workflow.add_operation(FilterIntensityRelative::new(0.01));
        workflow.add_operation(SquareRootTransformIntensity::new());
        workflow.add_operation(KeepTopNPeaks::new(50));
        workflow.add_operation(MinNumberOfPeaks::new(5));
        workflow.add_operation(MinMZRangeFilter::new(250.0));

        Self {
            workflow,
            filter_unknown_charge: true,
            ..Default::default()
        }
    }

    pub fn calculate_n10_value(&self, clusters: &FlatClusters) -> usize {
        let mut cluster_sizes: Vec<usize> = clusters.iter().map(|cluster| cluster.len()).collect();
        cluster_sizes.sort_by(|a, b| b.cmp(a));
        let mut cumulator: usize = 0;
        for cluster_size in cluster_sizes {
            cumulator += cluster_size;
            if cumulator >= (self.total_spectra_ingested * 10) / 100 {
                return cluster_size;
            }
        }
        0
    }

    pub fn calculate_cluster_purities(
        &self,
        clusters: &FlatClusters,
        mz_tol: f64,
        rt_tol: f32,
    ) -> Vec<f64> {
        clusters
            .iter()
            .map(|cluster| self.calculate_cluster_purity(cluster, mz_tol, rt_tol))
            .collect()
    }

    pub fn average_weighted_cluster_purity(
        &self,
        clusters: &FlatClusters,
        mz_tol: f64,
        rt_tol: f32,
    ) -> f64 {
        let purities = self.calculate_cluster_purities(clusters, mz_tol, rt_tol);

        let (product, sum) = clusters
            .iter()
            .zip(purities)
            .fold((0.0, 0), |acc, (cluster, purity)| {
                (acc.0 + cluster.len() as f64 * purity, acc.1 + cluster.len())
            });
        product / sum as f64
    }

    pub fn average_cluster_purity(&self, clusters: &FlatClusters, mz_tol: f64, rt_tol: f32) -> f64 {
        let purities = self.calculate_cluster_purities(clusters, mz_tol, rt_tol);

        let sum = purities.iter().sum::<f64>();
        sum / purities.len() as f64
    }

    pub fn calculate_cluster_purity(&self, cluster_ids: &[usize], mz_tol: f64, rt_tol: f32) -> f64 {
        let n = cluster_ids.len();
        if n <= 1 {
            return 1.0;
        }

        let mut parent: Vec<usize> = (0..n).collect();
        let mut component_size: Vec<usize> = vec![1; n];

        fn find(i: usize, parent: &mut Vec<usize>) -> usize {
            if parent[i] == i {
                i
            } else {
                parent[i] = find(parent[i], parent);
                parent[i]
            }
        }

        fn unite(i: usize, j: usize, parent: &mut Vec<usize>, size: &mut Vec<usize>) {
            let root_i = find(i, parent);
            let root_j = find(j, parent);
            if root_i != root_j {
                parent[root_i] = root_j;
                size[root_j] += size[root_i];
            }
        }

        let mut file_assignments = Vec::with_capacity(n);
        for &id in cluster_ids {
            let f_idx = self.metadata.get(id).unwrap().file_index;
            file_assignments.push(f_idx);
        }

        for i in 0..n {
            for j in i + 1..n {
                if file_assignments[i] == file_assignments[j] {
                    let s1 = self.metadata.get(cluster_ids[i]).unwrap();
                    let s2 = self.metadata.get(cluster_ids[j]).unwrap();

                    let mz_diff = (s1.precursor_mz - s2.precursor_mz).abs();
                    let rt_diff =
                        (s1.retention_time_in_seconds - s2.retention_time_in_seconds).abs();

                    if mz_diff <= mz_tol && rt_diff <= rt_tol {
                        unite(i, j, &mut parent, &mut component_size);
                    }
                }
            }
        }

        let mut file_max_components = vec![0usize; self.file_names.len()];

        for i in 0..n {
            let root = find(i, &mut parent);
            let f_idx = file_assignments[i];
            let size = component_size[root];

            if size > file_max_components[f_idx] {
                file_max_components[f_idx] = size;
            }
        }

        let sum_max_components: usize = file_max_components.iter().sum();

        sum_max_components as f64 / n as f64
    }

    pub fn ingest_files_parallel(&mut self, folder_path: &str) {
        self.workflow.use_parallel = false;
        let paths: Vec<_> = std::fs::read_dir(folder_path)
            .unwrap()
            .filter_map(|e| {
                let path = e.ok()?.path();
                let s = path.to_str()?.to_string();
                if s.ends_with(".mgf") { Some(s) } else { None }
            })
            .collect();

        let total_disk_size: u64 = paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();

        let estimated_total_spectra = (total_disk_size / MGF_SPECTRA_SIZE as u64) as usize;

        self.spectra.reserve(estimated_total_spectra);
        self.metadata.reserve(estimated_total_spectra);

        let start_index = self.file_names.len();

        self.file_names.extend(paths.iter().map(|p| {
            p.split('/')
                .last()
                .unwrap()
                .split(".mgf")
                .next()
                .unwrap()
                .to_string()
        }));

        let start_time = std::time::Instant::now();
        let results: Vec<(Vec<MSNSpectrum>, Vec<SpectrumMetadata>, usize)> = paths
            .into_par_iter()
            .enumerate()
            .map(|(i, path)| {
                let current_file_index = start_index + i;
                self.process_single_file(&path, current_file_index)
            })
            .collect();

        for (mut file_spectra, mut file_metadata, raw_count) in results {
            self.total_spectra_ingested += raw_count;
            self.metadata.append(&mut file_metadata);
            self.spectra.append(&mut file_spectra);
        }
        self.spectra
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, spectrum)| spectrum.id = index as u64);
    }

    fn process_single_file(
        &self,
        path: &str,
        file_index: usize,
    ) -> (Vec<MSNSpectrum>, Vec<SpectrumMetadata>, usize) {
        let mut parser = MgfParser::default();
        parser.config.use_parallel_parsing = false;
        let mut spectra = parser.parse_file(path).unwrap();

        let raw_count = spectra.len();
        self.workflow.execute(&mut spectra);

        let mut local_metadata = Vec::new();
        let processed_spectra: Vec<MSNSpectrum> = spectra
            .into_iter()
            .filter(|s| {
                !s.peaks().is_empty()
                    && if self.filter_unknown_charge {
                        s.get_charge() != 0
                    } else {
                        true
                    }
            })
            .enumerate()
            .map(|(local_idx, s)| {
                let scan_number = s
                    .get_metadata("SCANS")
                    .unwrap_or("0")
                    .parse::<usize>()
                    .unwrap_or(0);
                let mut msn = s.to_msn();

                msn.id = local_idx as u64;

                local_metadata.push(SpectrumMetadata {
                    precursor_mz: msn.precursor_mz,
                    retention_time_in_seconds: msn.retention_time_in_seconds,
                    charge: msn.charge(),
                    scan_number,
                    file_index,
                });

                msn
            })
            .collect();

        (processed_spectra, local_metadata, raw_count)
    }

    pub fn ingest_file(&mut self, file_path: &str) {
        let file_index = self.file_names.len();
        self.file_names.push(
            file_path
                .split("/")
                .last()
                .unwrap()
                .split(".mgf")
                .next()
                .unwrap()
                .to_string(),
        );

        let mut spectra = self.parser.parse_file(file_path).unwrap();
        self.total_spectra_ingested += spectra.len();
        self.workflow.execute(&mut spectra);

        let msn_spectra: Vec<MSNSpectrum> = spectra
            .into_iter()
            .filter(|s| {
                !(s.peaks().len() == 0)
                    && if self.filter_unknown_charge {
                        s.get_charge() != 0
                    } else {
                        true
                    }
            })
            .map(|s| {
                (
                    s.get_metadata("SCANS")
                        .unwrap_or("0")
                        .parse::<usize>()
                        .unwrap_or(0),
                    s.to_msn(),
                )
            })
            .inspect(|(scan_number, msn)| {
                self.metadata.insert(
                    msn.id as usize,
                    SpectrumMetadata {
                        precursor_mz: msn.precursor_mz,
                        retention_time_in_seconds: msn.retention_time_in_seconds,
                        charge: msn.charge(),
                        scan_number: *scan_number,
                        file_index,
                    },
                );
            })
            .map(|(_, msn_spectrum)| msn_spectrum)
            .collect();

        self.spectra.extend(msn_spectra)
    }

    pub fn ingest_files(&mut self, folder_path: &str) {
        for file_entry in std::fs::read_dir(folder_path).unwrap() {
            let file_path = file_entry.unwrap().path();
            if let Some(path_str) = file_path.to_str() {
                if !path_str.ends_with(".mgf") {
                    continue;
                }
                self.ingest_file(path_str);
            }
        }
    }

    pub fn prepare_clustering(&mut self) {
        let spectra = std::mem::take(&mut self.spectra);
        self.engine.split_on_mz_and_charge(spectra).unwrap();
    }

    pub fn cluster(&mut self) -> FlatClusters {
        let clusters = self.engine.cluster().unwrap();

        let mz_values: Vec<f64> = clusters
            .data
            .iter()
            .map(|id| self.metadata.get(*id).unwrap().precursor_mz)
            .collect();
        clusters.refine_mz_clusters(&mz_values, self.engine.mz_tolerance)
    }

    pub fn cluster_spectra(&mut self) -> FlatClusters {
        self.prepare_clustering();
        self.cluster()
    }

    pub fn write_results(&self, clusters: &FlatClusters, output_file_name: &str) {
        if clusters.total_elements() == 0 {
            return;
        }
        let mut file = std::fs::File::create(output_file_name).unwrap();
        let mut writer = std::io::BufWriter::new(&mut file);
        writer.write(format!("# RedshiftMS v0.1.0 {}\n# eps = {}\n# hash_len = {}\n# total_spectra_parsed = {}\nidentifier,precursor_charge,precursor_mz,retention_time,cluster\n", if self.engine.exact {"(exact)"} else {""},  self.engine.epsilon, self.engine.hash_size(), self.total_spectra_ingested).as_bytes())
            .unwrap();
        let mut counter = 0;
        for cluster in clusters.iter() {
            counter += 1;
            for spectrum in cluster {
                let metadata = self.metadata.get(*spectrum).unwrap();
                let precursor_charge = metadata.charge;
                let precursor_mz = metadata.precursor_mz;
                let retention_time = metadata.retention_time_in_seconds;
                writer
                    .write(
                        format!(
                            "mzspec:USI000000:{}:scan:{},{},{},{},{}\n",
                            &self.file_names[metadata.file_index],
                            metadata.scan_number,
                            precursor_charge,
                            precursor_mz,
                            retention_time,
                            counter,
                        )
                        .as_bytes(),
                    )
                    .unwrap();
            }
        }
    }
}

pub fn calculate_cluster_recall(exact_labels: &[ClusterLabel], lsh_labels: &[ClusterLabel]) -> f32 {
    assert_eq!(exact_labels.len(), lsh_labels.len());
    let n = exact_labels.len();

    let mut true_positives = 0;
    let mut false_negatives = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let exact_same = match (exact_labels[i], exact_labels[j]) {
                (ClusterLabel::Cluster(id1), ClusterLabel::Cluster(id2)) => id1 == id2,
                _ => false,
            };

            if exact_same {
                let lsh_same = match (lsh_labels[i], lsh_labels[j]) {
                    (ClusterLabel::Cluster(id1), ClusterLabel::Cluster(id2)) => id1 == id2,
                    _ => false,
                };

                if lsh_same {
                    true_positives += 1;
                } else {
                    false_negatives += 1;
                }
            }
        }
    }

    let total_true_pairs = true_positives + false_negatives;
    if total_true_pairs == 0 {
        return 1.0;
    }

    true_positives as f32 / total_true_pairs as f32
}
