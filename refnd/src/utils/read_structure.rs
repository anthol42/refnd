use std::path::Path;
use std::panic;
use pdbtbx::{StrictnessLevel, ReadOptions};
use crate::kernels::proteins::foldseek::{StructureData, encode_3di};

fn three_to_one(name: &str) -> char {
    match name {
        "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
        "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
        "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
        "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
        _ => 'X',
    }
}

/// Parse a PDB or mmCIF file and return a `StructureData` with the amino-acid
/// sequence and the corresponding 3Di structural alphabet sequence.
///
/// Only the first model and first chain are used. Residues missing CA, N or C
/// atoms are skipped. Missing CB atoms (e.g. GLY) are passed as NaN — the
/// 3Di encoder approximates them internally.
pub fn read_structure(path: &Path) -> Result<StructureData, String> {
    let path_str = path.to_str()
        .ok_or_else(|| format!("Non-UTF8 path: {}", path.display()))?;
    let path_owned = path_str.to_owned();
    let parse_result = panic::catch_unwind(|| {
        ReadOptions::default()
            .set_level(StrictnessLevel::Loose)
            .read(path_owned)
    });
    let (pdb, _) = parse_result
        .map_err(|_| format!("pdbtbx panicked on '{}'", path.display()))?
        .map_err(|e| format!("Failed to parse '{}': {e:?}", path.display()))?;

    let model = pdb.models().next()
        .ok_or_else(|| format!("No models in '{}'", path.display()))?;

    let mut aa_seq = String::new();
    let mut ca_x = Vec::new(); let mut ca_y = Vec::new(); let mut ca_z = Vec::new();
    let mut n_x  = Vec::new(); let mut n_y  = Vec::new(); let mut n_z  = Vec::new();
    let mut c_x  = Vec::new(); let mut c_y  = Vec::new(); let mut c_z  = Vec::new();
    let mut cb_x = Vec::new(); let mut cb_y = Vec::new(); let mut cb_z = Vec::new();

    // Use only the first chain
    if let Some(chain) = model.chains().next() {
        for residue in chain.residues() {
            let name = residue.name().unwrap_or("UNK");
            let ca = residue.atoms().find(|a| a.name() == "CA");
            let n  = residue.atoms().find(|a| a.name() == "N");
            let c  = residue.atoms().find(|a| a.name() == "C");
            let cb = residue.atoms().find(|a| a.name() == "CB");

            let (Some(ca), Some(n), Some(c)) = (ca, n, c) else { continue };

            aa_seq.push(three_to_one(name));
            ca_x.push(ca.x()); ca_y.push(ca.y()); ca_z.push(ca.z());
            n_x.push(n.x());   n_y.push(n.y());   n_z.push(n.z());
            c_x.push(c.x());   c_y.push(c.y());   c_z.push(c.z());
            let (cbx, cby, cbz) = cb
                .map(|a| (a.x(), a.y(), a.z()))
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
            cb_x.push(cbx); cb_y.push(cby); cb_z.push(cbz);
        }
    }

    if aa_seq.is_empty() {
        return Err(format!("No backbone atoms found in '{}'", path.display()));
    }

    let tdi_seq = encode_3di(
        &ca_x, &ca_y, &ca_z,
        &n_x,  &n_y,  &n_z,
        &c_x,  &c_y,  &c_z,
        &cb_x, &cb_y, &cb_z,
    );

    Ok(StructureData { aa_seq, tdi_seq })
}

/// Read all PDB/mmCIF files from a directory, sorted by filename.
/// Returns `(label, StructureData)` pairs; files that fail to parse are skipped
/// with a warning printed to stderr.
pub fn read_structure_dir(dir: &Path) -> Vec<(String, StructureData)> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Cannot read '{}': {e}", dir.display()))
        .filter_map(|e| {
            let path = e.ok()?.path();
            let ext = path.extension()?.to_str()?.to_lowercase();
            matches!(ext.as_str(), "pdb" | "cif" | "mmcif" | "ent").then_some(path)
        })
        .collect();
    paths.sort();

    paths
        .into_iter()
        .filter_map(|path| {
            let label = path.file_stem()?.to_string_lossy().into_owned();
            match read_structure(&path) {
                Ok(data) => Some((label, data)),
                Err(e) => {
                    eprintln!("warning: skipping {}: {e}", path.display());
                    None
                }
            }
        })
        .collect()
}
