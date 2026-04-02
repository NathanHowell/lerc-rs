// Expose Lerc2::ComputeChecksumFletcher32 (which is private) so we can
// property-test the Rust reimplementation against the C++ original.
//
// We use the `#define private public` trick to access the private static
// method without modifying the upstream header.

#define private public
#include "Lerc2.h"
#undef private

USING_NAMESPACE_LERC

extern "C" unsigned int lerc_fletcher32(const unsigned char* pByte, int len) {
    return Lerc2::ComputeChecksumFletcher32(pByte, len);
}
