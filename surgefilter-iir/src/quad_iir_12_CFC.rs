crate::ix!();

coeffidx!{
    C;
    Ar,
    Ai,
    B1,
    Unused,
    C1,
    C2,
    D,
    Clipgain
}

#[cfg(target_arch = "x86_64")] 
pub fn iir_12_cfc_quad(qfu: &mut QuadFilterUnitState, input: __m128) -> __m128 {

    unsafe {
        // State-space with clipgain (2nd order, limit within register)

        qfu.coeff[C::Ar] = _mm_add_ps(qfu.coeff[C::Ar], qfu.dcoeff[C::Ar]); // ar
        qfu.coeff[C::Ai] = _mm_add_ps(qfu.coeff[C::Ai], qfu.dcoeff[C::Ai]); // ai
        qfu.coeff[C::B1] = _mm_add_ps(qfu.coeff[C::B1], qfu.dcoeff[C::B1]); // b1
        qfu.coeff[C::C1] = _mm_add_ps(qfu.coeff[C::C1], qfu.dcoeff[C::C1]); // c1
        qfu.coeff[C::C2] = _mm_add_ps(qfu.coeff[C::C2], qfu.dcoeff[C::C2]); // c2
        qfu.coeff[C::D]  = _mm_add_ps(qfu.coeff[C::D],  qfu.dcoeff[C::D]); // d

        // y(i) = c1.*s(1) + c2.*s(2) + d.*x(i);
        // s1 = ar.*s(1) - ai.*s(2) + x(i);
        // s2 = ai.*s(1) + ar.*s(2);

        let y: __m128 = _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(qfu.coeff[C::C1], qfu.reg[C::Ar]),
                _mm_mul_ps(qfu.coeff[C::D], input)
            ),
            _mm_mul_ps( qfu.coeff[C::C2], qfu.reg[C::Ai])
        );

        let s1: __m128 = _mm_add_ps(
            _mm_mul_ps(input, qfu.coeff[C::B1]),
            _mm_sub_ps(
                _mm_mul_ps(qfu.coeff[C::Ar], qfu.reg[C::Ar]),
                _mm_mul_ps(qfu.coeff[C::Ai], qfu.reg[C::Ai])
            )
        );

        let s2: __m128 = _mm_add_ps(
            _mm_mul_ps(qfu.coeff[C::Ai], qfu.reg[C::Ar]), 
            _mm_mul_ps(qfu.coeff[C::Ar], qfu.reg[C::Ai])
        );

        qfu.reg[C::Ar] = _mm_mul_ps(s1, qfu.reg[C::B1]);
        qfu.reg[C::Ai] = _mm_mul_ps(s2, qfu.reg[C::B1]);

        // Clipgain
        qfu.coeff[C::Clipgain] = _mm_add_ps(qfu.coeff[C::Clipgain], qfu.dcoeff[C::Clipgain]); 

        let m01: __m128 = _mm_set1_ps(0.1);
        let m1:  __m128 = _mm_set1_ps(1.0);

        qfu.reg[C::B1] = _mm_max_ps(
            m01, 
            _mm_sub_ps(
                m1, 
                _mm_mul_ps(qfu.coeff[C::Clipgain], _mm_mul_ps(y, y))
            )
        );

        y
    }
}
