#include "wrapper.h"
#include "foldseek/lib/3di/structureto3di.h"
#include <cstdlib>
#include <vector>

// Maps state index 0-19 to the 3Di alphabet character; anything else -> 'X'
static const char STATE_CHARS[] = "ACDEFGHIKLMNPQRSTVWY";

extern "C" {

char* foldseek_encode_3di(
    const double* ca_x, const double* ca_y, const double* ca_z,
    const double* n_x,  const double* n_y,  const double* n_z,
    const double* c_x,  const double* c_y,  const double* c_z,
    const double* cb_x, const double* cb_y, const double* cb_z,
    int len
) {
    std::vector<Vec3> ca(len), n(len), c(len), cb(len);
    for (int i = 0; i < len; ++i) {
        ca[i] = { ca_x[i], ca_y[i], ca_z[i] };
        n[i]  = { n_x[i],  n_y[i],  n_z[i]  };
        c[i]  = { c_x[i],  c_y[i],  c_z[i]  };
        cb[i] = { cb_x[i], cb_y[i], cb_z[i] }; // NaN => approxCBetaPosition inside
    }

    // Loaded once per thread; model weights are embedded at compile time.
    static thread_local StructureTo3Di encoder;
    const char* states = encoder.structure2states(
        ca.data(), n.data(), c.data(), cb.data(), static_cast<size_t>(len)
    );

    char* result = static_cast<char*>(std::malloc(len + 1));
    for (int i = 0; i < len; ++i) {
        unsigned char s = static_cast<unsigned char>(states[i]);
        result[i] = (s < 20) ? STATE_CHARS[s] : 'X';
    }
    result[len] = '\0';
    return result;
}

void foldseek_free_str(char* s) {
    std::free(s);
}

} // extern "C"
