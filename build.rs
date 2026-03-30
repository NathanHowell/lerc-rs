fn main() {
    #[cfg(feature = "cpp-validation")]
    {
        cc::Build::new()
            .cpp(true)
            .std("c++14")
            .define("LERC_STATIC", None)
            .include("esri-lerc/src/LercLib/include")
            .include("esri-lerc/src/LercLib")
            // Core LercLib sources
            .file("esri-lerc/src/LercLib/BitMask.cpp")
            .file("esri-lerc/src/LercLib/BitStuffer2.cpp")
            .file("esri-lerc/src/LercLib/Huffman.cpp")
            .file("esri-lerc/src/LercLib/Lerc.cpp")
            .file("esri-lerc/src/LercLib/Lerc2.cpp")
            .file("esri-lerc/src/LercLib/Lerc_c_api_impl.cpp")
            .file("esri-lerc/src/LercLib/RLE.cpp")
            // FPL (Float Point Lossless) sources
            .file("esri-lerc/src/LercLib/fpl_Compression.cpp")
            .file("esri-lerc/src/LercLib/fpl_EsriHuffman.cpp")
            .file("esri-lerc/src/LercLib/fpl_Lerc2Ext.cpp")
            .file("esri-lerc/src/LercLib/fpl_Predictor.cpp")
            .file("esri-lerc/src/LercLib/fpl_UnitTypes.cpp")
            // Lerc1 decode support
            .file("esri-lerc/src/LercLib/Lerc1Decode/BitStuffer.cpp")
            .file("esri-lerc/src/LercLib/Lerc1Decode/CntZImage.cpp")
            .warnings(false)
            .compile("lerc_cpp");
    }
}
