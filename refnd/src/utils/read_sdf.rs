use crate::utils::fingerprints::BitFingerprint;
use clap::ValueEnum;
use fixedbitset::FixedBitSet;
use molprint_core::{MolGraph, smiles::parse_smiles};
use molprint_fp::{maccs::Maccs166, morgan::Morgan, traits::Fingerprinter};
use molprint_io::sdf::open_sdf;
use std::{fs, path::Path};

#[derive(ValueEnum, Clone, Debug)]
pub enum FingerprintType {
    Morgan,
    Maccs,
}

/// Reads a single SDF file. Returns (label, fingerprint) pairs where the label
/// is the molecule index within the file ("0", "1", …).
pub fn read_sdf(
    path: &Path,
    fp_type: &FingerprintType,
) -> Result<Vec<(String, BitFingerprint)>, String> {
    let mut entries = Vec::new();
    for (idx, result) in open_sdf(path).map_err(|e| e.to_string())?.enumerate() {
        let record = result.map_err(|e| e.to_string())?;
        entries.push((idx.to_string(), compute_fp(&record.mol, fp_type)));
    }
    Ok(entries)
}

/// Reads a SMILES file (one entry per line: `SMILES [name]`).
/// Returns (name_or_index, fingerprint) pairs.
pub fn read_smi(
    path: &Path,
    fp_type: &FingerprintType,
) -> Result<Vec<(String, BitFingerprint)>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut entries = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let smiles = parts.next().unwrap();
        let label = parts
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| idx.to_string());
        let mol = parse_smiles(smiles).map_err(|e| format!("line {idx}: {e}"))?;
        entries.push((label, compute_fp(&mol, fp_type)));
    }
    Ok(entries)
}

/// Reads a CSV file that has a `smiles` or `SMILES` column and uses the first
/// column as the label. Returns (label, fingerprint) pairs.
pub fn read_csv_mol(
    path: &Path,
    fp_type: &FingerprintType,
) -> Result<Vec<(String, BitFingerprint)>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines = content.lines();

    let header = lines.next().ok_or("CSV file is empty")?;
    let cols: Vec<&str> = header.split(',').collect();

    let smiles_col = cols
        .iter()
        .position(|h| h.trim().eq_ignore_ascii_case("smiles"))
        .ok_or("no `smiles` or `SMILES` column found in CSV")?;

    let mut entries = Vec::new();
    for (row_idx, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        let label = fields
            .first()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| row_idx.to_string());
        let smiles = fields
            .get(smiles_col)
            .ok_or_else(|| format!("row {row_idx}: missing SMILES column"))?
            .trim();
        let mol = parse_smiles(smiles).map_err(|e| format!("row {row_idx}: {e}"))?;
        entries.push((label, compute_fp(&mol, fp_type)));
    }
    Ok(entries)
}

/// Reads a molecule file, inferring the format from the extension.
/// Supported extensions: `.sdf`, `.smi` / `.smiles`, `.csv`.
pub fn read_molecule_file(
    path: &Path,
    fp_type: &FingerprintType,
) -> Result<Vec<(String, BitFingerprint)>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "sdf" => read_sdf(path, fp_type),
        "smi" | "smiles" => read_smi(path, fp_type),
        "csv" => read_csv_mol(path, fp_type),
        other => Err(format!(
            "unknown extension `.{other}` for `{}` — supported: .sdf, .smi, .smiles, .csv",
            path.display()
        )),
    }
}

fn compute_fp(mol: &MolGraph, fp_type: &FingerprintType) -> BitFingerprint {
    let fp = match fp_type {
        FingerprintType::Morgan => Morgan::new(2, 2048).fingerprint(mol),
        FingerprintType::Maccs => Maccs166::new().fingerprint(mol),
    };
    let mut fbs = FixedBitSet::with_capacity(fp.nbits());
    for i in 0..fp.nbits() {
        if fp.get(i) {
            fbs.set(i, true);
        }
    }
    BitFingerprint::new(fbs)
}
