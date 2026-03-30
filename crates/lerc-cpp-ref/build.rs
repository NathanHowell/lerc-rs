fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let lerc_lib = root.join("esri-lerc/src/LercLib");

    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .define("LERC_STATIC", None)
        .include(lerc_lib.join("include"))
        .include(&lerc_lib)
        // Core LercLib sources
        .file(lerc_lib.join("BitMask.cpp"))
        .file(lerc_lib.join("BitStuffer2.cpp"))
        .file(lerc_lib.join("Huffman.cpp"))
        .file(lerc_lib.join("Lerc.cpp"))
        .file(lerc_lib.join("Lerc2.cpp"))
        .file(lerc_lib.join("Lerc_c_api_impl.cpp"))
        .file(lerc_lib.join("RLE.cpp"))
        // FPL (Float Point Lossless) sources
        .file(lerc_lib.join("fpl_Compression.cpp"))
        .file(lerc_lib.join("fpl_EsriHuffman.cpp"))
        .file(lerc_lib.join("fpl_Lerc2Ext.cpp"))
        .file(lerc_lib.join("fpl_Predictor.cpp"))
        .file(lerc_lib.join("fpl_UnitTypes.cpp"))
        // Lerc1 decode support
        .file(lerc_lib.join("Lerc1Decode/BitStuffer.cpp"))
        .file(lerc_lib.join("Lerc1Decode/CntZImage.cpp"))
        .warnings(false)
        .compile("lerc_cpp");
}
