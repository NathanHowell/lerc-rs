#include <cstdio>
#include "Lerc_types.h"

// Print Rust constant definitions derived from the C++ headers.
int main() {
    using namespace LercNS;

    puts("// Auto-generated from Lerc_types.h — do not edit");
    puts("");

    // DataType
    puts("// DataType codes");
    printf("pub const DT_CHAR: u32 = %d;\n",   (int)DataType::dt_char);
    printf("pub const DT_UCHAR: u32 = %d;\n",  (int)DataType::dt_uchar);
    printf("pub const DT_SHORT: u32 = %d;\n",  (int)DataType::dt_short);
    printf("pub const DT_USHORT: u32 = %d;\n", (int)DataType::dt_ushort);
    printf("pub const DT_INT: u32 = %d;\n",    (int)DataType::dt_int);
    printf("pub const DT_UINT: u32 = %d;\n",   (int)DataType::dt_uint);
    printf("pub const DT_FLOAT: u32 = %d;\n",  (int)DataType::dt_float);
    printf("pub const DT_DOUBLE: u32 = %d;\n", (int)DataType::dt_double);
    puts("");

    // InfoArrOrder
    puts("// InfoArray indices");
    printf("pub const INFO_VERSION: usize = %d;\n",          (int)InfoArrOrder::version);
    printf("pub const INFO_DATA_TYPE: usize = %d;\n",        (int)InfoArrOrder::dataType);
    printf("pub const INFO_N_DEPTH: usize = %d;\n",          (int)InfoArrOrder::nDim);
    printf("pub const INFO_N_COLS: usize = %d;\n",           (int)InfoArrOrder::nCols);
    printf("pub const INFO_N_ROWS: usize = %d;\n",           (int)InfoArrOrder::nRows);
    printf("pub const INFO_N_BANDS: usize = %d;\n",          (int)InfoArrOrder::nBands);
    printf("pub const INFO_N_VALID_PIXELS: usize = %d;\n",   (int)InfoArrOrder::nValidPixels);
    printf("pub const INFO_BLOB_SIZE: usize = %d;\n",        (int)InfoArrOrder::blobSize);
    printf("pub const INFO_N_MASKS: usize = %d;\n",          (int)InfoArrOrder::nMasks);
    printf("pub const INFO_N_USES_NO_DATA: usize = %d;\n",   (int)InfoArrOrder::nUsesNoDataValue);
    printf("pub const INFO_ARRAY_SIZE: usize = %d;\n",       (int)InfoArrOrder::_last);
    puts("");

    // DataRangeArrOrder
    puts("// DataRangeArray indices");
    printf("pub const RANGE_Z_MIN: usize = %d;\n",           (int)DataRangeArrOrder::zMin);
    printf("pub const RANGE_Z_MAX: usize = %d;\n",           (int)DataRangeArrOrder::zMax);
    printf("pub const RANGE_MAX_Z_ERR_USED: usize = %d;\n",  (int)DataRangeArrOrder::maxZErrUsed);
    printf("pub const RANGE_ARRAY_SIZE: usize = %d;\n",      (int)DataRangeArrOrder::_last);
    puts("");

    // ErrCode
    puts("// Error codes");
    printf("pub const ERR_OK: u32 = %d;\n",              (int)ErrCode::Ok);
    printf("pub const ERR_FAILED: u32 = %d;\n",          (int)ErrCode::Failed);
    printf("pub const ERR_WRONG_PARAM: u32 = %d;\n",     (int)ErrCode::WrongParam);
    printf("pub const ERR_BUFFER_TOO_SMALL: u32 = %d;\n",(int)ErrCode::BufferTooSmall);
    printf("pub const ERR_NAN: u32 = %d;\n",             (int)ErrCode::NaN);
    printf("pub const ERR_HAS_NO_DATA: u32 = %d;\n",     (int)ErrCode::HasNoData);

    return 0;
}
