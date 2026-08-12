//! Portable floating-point helpers usable in both `std` and `no_std` builds.
//!
//! `core` doesn't provide `floor`/`round`/`log2` -- unlike `sqrt`, they
//! aren't guaranteed LLVM intrinsics, so `std` implements them by linking
//! against the platform's libm. Under `no_std` there is no such libm, so we
//! fall back to the pure-Rust `libm` crate instead.

#[inline]
pub(crate) fn floor(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floor(x)
    }
}

#[inline]
pub(crate) fn round(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.round()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::round(x)
    }
}

#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

#[inline]
pub(crate) fn log2(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.log2()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log2(x)
    }
}
