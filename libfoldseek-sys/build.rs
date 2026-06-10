use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // ── 1. Embed encoder weights as a C header ──────────────────────────────
    let weights_path = manifest.join("foldseek/data/encoder_weights_3di.kerasify");
    let weights = fs::read(&weights_path).unwrap_or_else(|_| {
        panic!(
            "Cannot read encoder_weights_3di.kerasify — \
             make sure the foldseek submodule is initialised:\n  \
             git submodule update --init libfoldseek-sys/foldseek"
        )
    });

    let mut header = String::with_capacity(weights.len() * 6 + 128);
    header.push_str("static const unsigned char encoder_weights_3di_kerasify[] = {\n");
    for chunk in weights.chunks(16) {
        for b in chunk {
            header.push_str(&format!("0x{b:02x},"));
        }
        header.push('\n');
    }
    header.push_str("};\n");
    header.push_str(&format!(
        "static const unsigned int encoder_weights_3di_kerasify_len = {};\n",
        weights.len()
    ));
    fs::write(out.join("encoder_weights_3di.kerasify.h"), &header).unwrap();

    // ── 2. Compile C++ sources ───────────────────────────────────────────────
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("wrapper.cpp")
        .file("foldseek/lib/3di/structureto3di.cpp")
        .file("foldseek/lib/kerasify/keras_model.cpp")
        .include(&manifest)                              // wrapper.h
        .include(manifest.join("foldseek/lib"))         // kerasify/keras_model.h
        .include(manifest.join("foldseek/lib/mmseqs/lib/simde")) // simde/simde-common.h
        .include(&out)                                  // encoder_weights_3di.kerasify.h
        .compile("foldseek3di");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-changed=foldseek/lib/3di/structureto3di.h");
    println!("cargo:rerun-if-changed=foldseek/lib/3di/structureto3di.cpp");
    println!("cargo:rerun-if-changed=foldseek/lib/kerasify/keras_model.cpp");
    println!("cargo:rerun-if-changed=foldseek/data/encoder_weights_3di.kerasify");

    // ── 3. Generate Rust bindings ────────────────────────────────────────────
    bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("foldseek_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed")
        .write_to_file(out.join("bindings.rs"))
        .expect("could not write bindings.rs");
}
