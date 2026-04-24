// This file is a simple collection of contaminants that ruin experiments, mainly plasticizers
use chemistry_consts::PROTON_MASS_AMU as PROTON_MASS;

// Exact neutral monoisotopic masses of the base molecules
pub const BASE_POS_CONTAMINANTS: [f64; 8] = [
    148.016045, // Phthalic anhydride (C8H4O3)
    166.026610, // Phthalic acid (C8H6O4)
    194.057910, // Dimethyl phthalate (C10H10O4)
    222.089210, // Diethyl phthalate (C12H14O4)
    278.151810, // Dibutyl phthalate (C16H22O4)
    312.136160, // Benzyl butyl phthalate (C19H20O4)
    390.277010, // Bis(2-ethylhexyl) phthalate (C24H38O4)
    418.308310, // Diisononyl phthalate (C26H42O4)
];

pub const POS_CONTAMINANTS: [f64; 8] = [
    BASE_POS_CONTAMINANTS[0] + PROTON_MASS, // Phthalic anhydride
    BASE_POS_CONTAMINANTS[1] + PROTON_MASS, // Phthalic acid
    BASE_POS_CONTAMINANTS[2] + PROTON_MASS, // Dimethyl phthalate
    BASE_POS_CONTAMINANTS[3] + PROTON_MASS, // Diethyl phthalate
    BASE_POS_CONTAMINANTS[4] + PROTON_MASS, // Dibutyl phthalate
    BASE_POS_CONTAMINANTS[5] + PROTON_MASS, // Benzyl butyl phthalate
    BASE_POS_CONTAMINANTS[6] + PROTON_MASS, // Bis(2-ethylhexyl) phthalate
    BASE_POS_CONTAMINANTS[7] + PROTON_MASS, // Diisononyl phthalate
];

// Exact neutral monoisotopic masses of base molecules for NEGATIVE mode
pub const BASE_NEG_CONTAMINANTS: [f64; 6] = [
    113.992864, // Trifluoroacetic acid (TFA, C2HF3O2)
    256.240230, // Palmitic acid (C16H32O2, common fatty acid)
    266.155181, // Dodecyl sulfate (from SDS, C12H26O4S)
    282.255880, // Oleic acid (C18H34O2, skin/oil fatty acid)
    284.271530, // Stearic acid (C18H36O2, common fatty acid)
    166.026610, // Phthalic acid (C8H6O4)
];

pub const NEG_CONTAMINANTS: [f64; 6] = [
    BASE_NEG_CONTAMINANTS[0] - PROTON_MASS, // TFA
    BASE_NEG_CONTAMINANTS[1] - PROTON_MASS, // Palmitic acid
    BASE_NEG_CONTAMINANTS[2] - PROTON_MASS, // Dodecyl sulfate
    BASE_NEG_CONTAMINANTS[3] - PROTON_MASS, // Oleic acid
    BASE_NEG_CONTAMINANTS[4] - PROTON_MASS, // Stearic acid
    BASE_NEG_CONTAMINANTS[5] - PROTON_MASS, // Phthalic acid
];

#[test]
fn test_proton_mass() {
    println!("PROTON MASS: {}", PROTON_MASS)
}
