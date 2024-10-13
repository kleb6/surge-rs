crate::ix!();

/// Sums the elements of a __m128 vector and returns the
/// result as a f32 value.
///
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn v_sum(x: __m128) -> f32 {
    unsafe {
        let mut f: f32 = 0.0;

        let mut a: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));

        a = _mm_add_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE!(0, 0, 0, 1)));

        _mm_store_ss(&mut f, a);

        f
    }
}
