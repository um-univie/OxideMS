//! ClassyFire Chemical Taxonomy Ontology definitions.
//!
//! Provides hierarchical ontology terms (Kingdom -> Superclass -> Class -> Subclass)
//! with official CHEMONTIDs and taxonomic parent-child relationships.
//!
//! The nodes are a subset of ChemOnt, the ClassyFire chemical taxonomy:
//! Djoumbou Feunang, Y. et al. "ClassyFire: automated chemical classification
//! with a comprehensive, computable taxonomy." J. Cheminform. 8, 61 (2016).
//! <https://doi.org/10.1186/s13321-016-0174-y>

/// Taxonomic level in the ClassyFire chemical ontology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaxonomyLevel {
    Kingdom = 1,
    Superclass = 2,
    Class = 3,
    Subclass = 4,
    DirectParent = 5,
}

/// A node in the ClassyFire chemical taxonomy tree.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassyFireNode {
    /// Unique CHEMONT identifier (e.g. "CHEMONTID:0000010")
    pub chemont_id: &'static str,
    /// Human-readable category name (e.g. "Flavonoids", "Organoheterocyclic compounds")
    pub name: &'static str,
    /// Taxonomic level (Kingdom, Superclass, Class, Subclass)
    pub level: TaxonomyLevel,
    /// Parent node CHEMONT identifier in the ontology tree (None for Kingdom)
    pub parent_chemont_id: Option<&'static str>,
    /// Brief description / definition of the chemotype
    pub description: &'static str,
    /// Associated key MACCS key indices (0-165) that strongly support this chemotype
    pub key_maccs: &'static [usize],
    /// Prior background probability of this chemotype in natural products / metabolomics
    pub prior: f32,
}

/// Complete curated ClassyFire chemical taxonomy containing major Superclasses, Classes, and Subclasses
/// encompassing natural products, drugs, lipids, and specialized metabolites.
pub static CLASSYFIRE_TAXONOMY: &[ClassyFireNode] = &[
    // ══════════════════════════════════════════════════════════════════════════
    // Level 1: Kingdom
    // ══════════════════════════════════════════════════════════════════════════
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000000",
        name: "Organic compounds",
        level: TaxonomyLevel::Kingdom,
        parent_chemont_id: None,
        description: "Compounds containing at least one carbon atom, excluding elemental carbon, carbonates, and cyanides.",
        key_maccs: &[164, 160, 149],
        prior: 0.98,
    },

    // ══════════════════════════════════════════════════════════════════════════
    // Level 2: Superclasses
    // ══════════════════════════════════════════════════════════════════════════
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000012",
        name: "Lipids and lipid-like molecules",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Hydrophobic or amphipathic small molecules including fatty acyls, glycerolipids, sphingolipids, sterols, and prenols.",
        key_maccs: &[154, 158, 160, 162],
        prior: 0.25,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000284",
        name: "Phenylpropanoids and polyketides",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Organic compounds synthesized from phenylpropanoid precursors or polyketide synthases (flavonoids, isoflavonoids, coumarins, stilbenes).",
        key_maccs: &[144, 145, 139, 131],
        prior: 0.20,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000002",
        name: "Organoheterocyclic compounds",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Compounds containing a ring structure made up of carbon and at least one other element (N, O, S).",
        key_maccs: &[142, 138, 136, 120],
        prior: 0.35,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0002448",
        name: "Benzenoids",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Aromatic compounds containing one or more benzene rings.",
        key_maccs: &[136, 144, 162, 163],
        prior: 0.30,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000264",
        name: "Organic acids and derivatives",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Compounds containing one or more carboxylic acid groups or derivatives (amino acids, esters, amides).",
        key_maccs: &[141, 137, 127, 107],
        prior: 0.22,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000323",
        name: "Organic oxygen compounds",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Compounds containing carbon and oxygen, including carbohydrates, polyols, ethers, and aldehydes.",
        key_maccs: &[139, 143, 145, 147],
        prior: 0.18,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000278",
        name: "Organic nitrogen compounds",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Compounds containing carbon and nitrogen atoms (amines, quaternary ammonium, imines).",
        key_maccs: &[142, 133, 114, 88],
        prior: 0.20,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000347",
        name: "Nucleosides, nucleotides, and analogues",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Compounds composed of a purine or pyrimidine nucleobase linked to a ribose or deoxyribose sugar.",
        key_maccs: &[138, 143, 110, 84],
        prior: 0.05,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000014",
        name: "Alkaloids and derivatives",
        level: TaxonomyLevel::Superclass,
        parent_chemont_id: Some("CHEMONTID:0000000"),
        description: "Naturally occurring basic nitrogenous organic compounds derived from amino acids or specialized secondary metabolism.",
        key_maccs: &[142, 136, 138, 114],
        prior: 0.15,
    },

    // ══════════════════════════════════════════════════════════════════════════
    // Level 3: Classes
    // ══════════════════════════════════════════════════════════════════════════
    // --- Under Phenylpropanoids & polyketides ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000019",
        name: "Flavonoids",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000284"),
        description: "Secondary metabolites with a 15-carbon skeleton consisting of two phenyl rings and a heterocyclic ring (C6-C3-C6).",
        key_maccs: &[144, 145, 139, 131, 129],
        prior: 0.08,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000020",
        name: "Isoflavonoids",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000284"),
        description: "Flavonoid isomers containing a 3-phenylchromen-4-one backbone.",
        key_maccs: &[144, 145, 139, 129],
        prior: 0.03,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000027",
        name: "Coumarins and derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000284"),
        description: "Compounds containing a 2H-chromen-2-one (benzopyrone) ring system.",
        key_maccs: &[144, 139, 131, 115],
        prior: 0.04,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000030",
        name: "Stilbenes",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000284"),
        description: "Molecules with a 1,2-diphenylethylene core (e.g. resveratrol).",
        key_maccs: &[144, 136, 160, 162],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000024",
        name: "Anthocyanins",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000284"),
        description: "Water-soluble flavonoid pigments based on the flavylium (2-phenylchromenylium) cation.",
        key_maccs: &[144, 145, 139, 143],
        prior: 0.02,
    },

    // --- Under Lipids and lipid-like molecules ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000017",
        name: "Fatty Acyls",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000012"),
        description: "Aliphatic monocarboxylic acids with long hydrocarbon chains and their esters/amides.",
        key_maccs: &[154, 158, 160, 141],
        prior: 0.10,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000045",
        name: "Prenol lipids",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000012"),
        description: "Terpenoids and isoprenoids synthesized from 5-carbon isoprene units (mono-, sesqui-, di-, tri-terpenes).",
        key_maccs: &[160, 154, 149, 126],
        prior: 0.09,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000044",
        name: "Steroids and steroid derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000012"),
        description: "Lipids containing a cyclopentanoperhydrophenanthrene 4-ring core.",
        key_maccs: &[160, 158, 149, 139],
        prior: 0.05,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000037",
        name: "Glycerophospholipids",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000012"),
        description: "Glycerol-based phospholipids (PC, PE, PS, PI, PG, PA).",
        key_maccs: &[154, 158, 141, 110],
        prior: 0.04,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000038",
        name: "Sphingolipids",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000012"),
        description: "Lipids containing an amino alcohol sphingoid base.",
        key_maccs: &[154, 158, 142, 139],
        prior: 0.03,
    },

    // --- Under Organoheterocyclic compounds ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000078",
        name: "Indoles and derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000002"),
        description: "Heterocyclic compounds containing a fused benzene and pyrrole ring (e.g. tryptophan, melatonin, serotonin).",
        key_maccs: &[142, 136, 138, 120],
        prior: 0.06,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000074",
        name: "Pyridines and derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000002"),
        description: "Aromatic 6-membered heterocycles with one nitrogen atom (e.g. nicotinamide).",
        key_maccs: &[142, 138, 120, 98],
        prior: 0.07,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000085",
        name: "Quinolines and derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000002"),
        description: "Heterocyclic aromatic compounds with a fused benzene and pyridine ring.",
        key_maccs: &[142, 136, 138, 120],
        prior: 0.04,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000086",
        name: "Isoquinolines and derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000002"),
        description: "Heterocyclic aromatic compounds containing a benzo[c]pyridine core (e.g. morphine, glaucine, berberine).",
        key_maccs: &[142, 136, 138, 139],
        prior: 0.04,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000079",
        name: "Purines and purine derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000002"),
        description: "Fused heterocyclic aromatic compounds containing pyrimidine and imidazole rings (adenine, guanine, caffeine).",
        key_maccs: &[138, 142, 110, 84],
        prior: 0.03,
    },

    // --- Under Organic acids and derivatives ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000057",
        name: "Carboxylic acids and derivatives",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000264"),
        description: "Compounds containing a carboxyl group or its acyl derivatives.",
        key_maccs: &[141, 137, 127, 107],
        prior: 0.15,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000058",
        name: "Amino acids, peptides, and analogues",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000264"),
        description: "Compounds containing both an amino and a carboxyl functional group, as well as oligopeptides.",
        key_maccs: &[141, 142, 127, 88],
        prior: 0.10,
    },

    // --- Under Organic oxygen compounds ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000049",
        name: "Carbohydrates and carbohydrate conjugates",
        level: TaxonomyLevel::Class,
        parent_chemont_id: Some("CHEMONTID:0000323"),
        description: "Polyhydroxy aldehydes/ketones, monosaccharides, disaccharides, and their glycoside conjugates.",
        key_maccs: &[143, 139, 145, 147],
        prior: 0.08,
    },

    // ══════════════════════════════════════════════════════════════════════════
    // Level 4: Subclasses
    // ══════════════════════════════════════════════════════════════════════════
    // --- Flavonoid Subclasses ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000330",
        name: "Flavones",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000019"),
        description: "Flavonoids with a 2-phenylchromen-4-one core (e.g. luteolin, apigenin, tricetin, wogonin).",
        key_maccs: &[144, 145, 139, 131, 129],
        prior: 0.03,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000331",
        name: "Flavonols",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000019"),
        description: "3-hydroxyflavones (e.g. quercetin, kaempferol, myricetin, rutin).",
        key_maccs: &[144, 145, 139, 143, 131],
        prior: 0.03,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000332",
        name: "Flavanones",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000019"),
        description: "Saturated flavan-4-ones (e.g. naringenin, hesperetin).",
        key_maccs: &[144, 139, 131, 126],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000333",
        name: "Flavan-3-ols (Catechins)",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000019"),
        description: "Flavan-3-ol derivatives (e.g. catechin, epicatechin, EGCG).",
        key_maccs: &[144, 139, 143, 126],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000334",
        name: "O-methylated flavonoids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000019"),
        description: "Flavonoids containing one or more methoxy groups attached to the aromatic core (e.g. tangeritin, nobiletin, zapotin).",
        key_maccs: &[144, 139, 131, 147],
        prior: 0.03,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000335",
        name: "Flavonoid glycosides",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000019"),
        description: "Flavonoids covalently bonded to one or more sugar moieties (e.g. tiliroside, rutin, hesperidin).",
        key_maccs: &[144, 139, 143, 145],
        prior: 0.04,
    },

    // --- Terpenoid Subclasses ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000340",
        name: "Monoterpenoids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000045"),
        description: "C10 isoprenoid derivatives (limonene, menthol, camphor, pinene).",
        key_maccs: &[160, 149, 126],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000341",
        name: "Sesquiterpenoids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000045"),
        description: "C15 isoprenoid derivatives (artemisinin, caryophyllene).",
        key_maccs: &[160, 154, 149],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000342",
        name: "Diterpenoids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000045"),
        description: "C20 isoprenoid derivatives (paclitaxel, stevioside, gibberellins).",
        key_maccs: &[160, 154, 149, 139],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000343",
        name: "Triterpenoids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000045"),
        description: "C30 isoprenoid derivatives (oleanolic acid, ursolic acid, ginsenosides, squalene).",
        key_maccs: &[160, 158, 154, 141],
        prior: 0.03,
    },

    // --- Alkaloid Subclasses ---
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000350",
        name: "Aporphine alkaloids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000086"),
        description: "Tetracyclic isoquinoline alkaloids with a dibenzo[de,g]quinoline core (e.g. glaucine, boldine).",
        key_maccs: &[142, 136, 138, 139, 147],
        prior: 0.02,
    },
    ClassyFireNode {
        chemont_id: "CHEMONTID:0000351",
        name: "Indole alkaloids",
        level: TaxonomyLevel::Subclass,
        parent_chemont_id: Some("CHEMONTID:0000078"),
        description: "Alkaloids incorporating an indole nucleus (e.g. harmine, vinblastine, reserpine).",
        key_maccs: &[142, 136, 138, 120],
        prior: 0.03,
    },
];

/// Returns a reference to a ClassyFire node by its CHEMONT identifier.
pub fn find_node_by_chemont_id(id: &str) -> Option<&'static ClassyFireNode> {
    CLASSYFIRE_TAXONOMY.iter().find(|n| n.chemont_id == id)
}

/// Returns all child nodes of a given parent CHEMONT identifier.
pub fn get_children_of(parent_id: &str) -> Vec<&'static ClassyFireNode> {
    CLASSYFIRE_TAXONOMY
        .iter()
        .filter(|n| n.parent_chemont_id == Some(parent_id))
        .collect()
}
