// Expose Lerc2::ComputeChecksumFletcher32 (which is private) so we can
// property-test the Rust reimplementation against the C++ original.
//
// build.rs patches Lerc2.h in OUT_DIR to make the method public, and
// the include path is set so this file picks up the patched copy.

#include "Lerc2.h"

USING_NAMESPACE_LERC

extern "C" unsigned int lerc_fletcher32(const unsigned char* pByte, int len) {
    return Lerc2::ComputeChecksumFletcher32(pByte, len);
}
