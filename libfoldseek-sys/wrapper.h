#ifndef FOLDSEEK_WRAPPER_H
#define FOLDSEEK_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Encode backbone coordinates into a null-terminated 3Di sequence string.
 *
 * Each array has `len` elements (one per residue).
 * CB coordinates may be NaN — missing CB atoms (e.g. GLY) are approximated
 * internally from CA, N, C.
 *
 * Returns a heap-allocated string of length `len` using the 3Di alphabet
 * (ACDEFGHIKLMNPQRSTVWY; 'X' for invalid/terminal residues).
 * The caller must release it with foldseek_free_str(). */
char* foldseek_encode_3di(
    const double* ca_x, const double* ca_y, const double* ca_z,
    const double* n_x,  const double* n_y,  const double* n_z,
    const double* c_x,  const double* c_y,  const double* c_z,
    const double* cb_x, const double* cb_y, const double* cb_z,
    int len
);

void foldseek_free_str(char* s);

#ifdef __cplusplus
}
#endif
#endif
