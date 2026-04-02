use std::path::PathBuf;

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

    // --- Step 2: Compile the C++ LercLib as a static library ---

    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .define("LERC_STATIC", None)
        .include(lerc_lib.join("include"))
        .include(&lerc_lib)
        .file(lerc_lib.join("BitMask.cpp"))
        .file(lerc_lib.join("BitStuffer2.cpp"))
        .file(lerc_lib.join("Huffman.cpp"))
        .file(lerc_lib.join("Lerc.cpp"))
        .file(lerc_lib.join("Lerc2.cpp"))
        .file(lerc_lib.join("Lerc_c_api_impl.cpp"))
        .file(lerc_lib.join("RLE.cpp"))
        .file(lerc_lib.join("fpl_Compression.cpp"))
        .file(lerc_lib.join("fpl_EsriHuffman.cpp"))
        .file(lerc_lib.join("fpl_Lerc2Ext.cpp"))
        .file(lerc_lib.join("fpl_Predictor.cpp"))
        .file(lerc_lib.join("fpl_UnitTypes.cpp"))
        .file(lerc_lib.join("Lerc1Decode/BitStuffer.cpp"))
        .file(lerc_lib.join("Lerc1Decode/CntZImage.cpp"))
        .warnings(false)
        .compile("lerc_cpp");

    // Fletcher-32 shim: calls through to Lerc2::ComputeChecksumFletcher32
    let shim_src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fletcher32_shim.cpp");
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .define("LERC_STATIC", None)
        .include(lerc_lib.join("include"))
        .include(&lerc_lib)
        .file(&shim_src)
        .warnings(false)
        .compile("fletcher32_shim");
}
