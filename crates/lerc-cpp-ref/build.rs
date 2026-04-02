use std::path::{Path, PathBuf};

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let dest = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let lerc_lib = root.join("esri-lerc/src/LercLib");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // --- Step 1: Compile and run gen_constants.cpp to produce constants.rs ---

    let gen_src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("gen_constants.cpp");
    let gen_bin = out_dir.join("gen_constants");

    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let status = std::process::Command::new(&cxx)
        .args(["-std=c++14", "-o"])
        .arg(&gen_bin)
        .arg(&gen_src)
        .arg(format!("-I{}", lerc_lib.join("include").display()))
        .status()
        .expect("failed to compile gen_constants.cpp");
    assert!(status.success(), "gen_constants.cpp compilation failed");

    let output = std::process::Command::new(&gen_bin)
        .output()
        .expect("failed to run gen_constants");
    assert!(output.status.success(), "gen_constants exited with error");

    std::fs::write(out_dir.join("constants.rs"), &output.stdout)
        .expect("failed to write constants.rs");

    // --- Step 2: Copy LercLib to OUT_DIR and patch Lerc2.h ---
    // MSVC encodes access specifiers in mangled names, so Lerc2.cpp must be
    // compiled with the patched header. Quoted includes resolve relative to the
    // source file first, so we copy the entire tree to OUT_DIR and patch there.

    let patched_lib = out_dir.join("LercLib");
    let _ = std::fs::remove_dir_all(&patched_lib);
    copy_dir(&lerc_lib, &patched_lib);

    let header = std::fs::read_to_string(patched_lib.join("Lerc2.h")).unwrap();
    let patched = header.replace(
        "static unsigned int ComputeChecksumFletcher32(const Byte* pByte, int len);",
        "public: static unsigned int ComputeChecksumFletcher32(const Byte* pByte, int len); private:",
    );
    assert_ne!(header, patched, "patch target not found in Lerc2.h");
    std::fs::write(patched_lib.join("Lerc2.h"), patched).unwrap();

    // --- Step 3: Compile the patched LercLib and fletcher32 shim ---

    let shim_src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fletcher32_shim.cpp");
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .define("LERC_STATIC", None)
        .include(patched_lib.join("include"))
        .include(&patched_lib)
        .file(patched_lib.join("BitMask.cpp"))
        .file(patched_lib.join("BitStuffer2.cpp"))
        .file(patched_lib.join("Huffman.cpp"))
        .file(patched_lib.join("Lerc.cpp"))
        .file(patched_lib.join("Lerc2.cpp"))
        .file(patched_lib.join("Lerc_c_api_impl.cpp"))
        .file(patched_lib.join("RLE.cpp"))
        .file(patched_lib.join("fpl_Compression.cpp"))
        .file(patched_lib.join("fpl_EsriHuffman.cpp"))
        .file(patched_lib.join("fpl_Lerc2Ext.cpp"))
        .file(patched_lib.join("fpl_Predictor.cpp"))
        .file(patched_lib.join("fpl_UnitTypes.cpp"))
        .file(patched_lib.join("Lerc1Decode/BitStuffer.cpp"))
        .file(patched_lib.join("Lerc1Decode/CntZImage.cpp"))
        .file(&shim_src)
        .warnings(false)
        .compile("lerc_cpp");
}
