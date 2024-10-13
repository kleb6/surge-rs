crate::ix!();

#[inline]
pub fn v_sqrt_fast(v: __m128) -> __m128 {
    unsafe { _mm_rcp_ps(_mm_rsqrt_ps(v)) }
}

#[inline]
pub fn square(x: f64) -> f64 {
    x * x
}

#[cfg(target_arch = "x86_64")]
pub fn get_squaremax<NQ>(d: *mut f32, nquads: NQ) -> f32
where
    <NQ as std::convert::TryInto<u32>>::Error: std::fmt::Debug,
    NQ: TryInto<u32>,
{
    let nquads: u32 = nquads.try_into().unwrap();
    unsafe {
        let mut mx1: __m128 = _mm_setzero_ps();
        let mut mx2: __m128 = _mm_setzero_ps();

        for i in (0..nquads).step_by(2) {
            mx1 = _mm_max_ps(
                mx1,
                _mm_mul_ps(
                    *(d as *mut __m128).offset(i as isize),
                    *(d as *mut __m128).offset(i as isize),
                ),
            );
            mx2 = _mm_max_ps(
                mx2,
                _mm_mul_ps(
                    *(d as *mut __m128).offset((i + 1) as isize),
                    *(d as *mut __m128).offset((i + 1) as isize),
                ),
            );
        }

        mx1 = _mm_max_ps(mx1, mx2);
        mx1 = max_ps_to_ss(mx1);

        let mut f: f32 = 0.0;

        _mm_store_ss(&mut f, mx1);

        f
    }
}
