use libfoldseek_sys::{foldseek_encode_3di, foldseek_free_str};
use std::ffi::CStr;

#[derive(Clone)]
pub struct StructureData {
    pub aa_seq: String,
    pub tdi_seq: String,
}

/// Encode backbone coordinates into a 3Di sequence string.
///
/// Each slice must have the same length (one entry per residue).
/// Pass `f64::NAN` for missing CB atoms (e.g. GLY) — the encoder approximates
/// the CB position from CA, N, C internally.
pub fn encode_3di(
    ca_x: &[f64], ca_y: &[f64], ca_z: &[f64],
    n_x:  &[f64], n_y:  &[f64], n_z:  &[f64],
    c_x:  &[f64], c_y:  &[f64], c_z:  &[f64],
    cb_x: &[f64], cb_y: &[f64], cb_z: &[f64],
) -> String {
    let len = ca_x.len() as i32;
    let ptr = unsafe {
        foldseek_encode_3di(
            ca_x.as_ptr(), ca_y.as_ptr(), ca_z.as_ptr(),
            n_x.as_ptr(),  n_y.as_ptr(),  n_z.as_ptr(),
            c_x.as_ptr(),  c_y.as_ptr(),  c_z.as_ptr(),
            cb_x.as_ptr(), cb_y.as_ptr(), cb_z.as_ptr(),
            len,
        )
    };
    let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
    unsafe { foldseek_free_str(ptr) };
    s
}
