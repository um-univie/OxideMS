//! CANOPUS: Computational Compound Classification from Fragmentation Trees & Fingerprints.
//!
//! Predicts ClassyFire taxonomic chemical classifications (Kingdom, Superclass, Class, Subclass)
//! directly from predicted molecular fingerprint probability vectors.

pub mod classifier;
pub mod ontology;

pub use classifier::{CanopusClassifier, ClassyFireHit, CompoundClassification};
pub use ontology::{find_node_by_chemont_id, get_children_of, ClassyFireNode, TaxonomyLevel, CLASSYFIRE_TAXONOMY};
