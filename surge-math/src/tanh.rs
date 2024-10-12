crate::ix!();

#[allow(clippy::many_single_char_names)]
#[cfg(target_arch = "x86_64")] #[inline]
pub fn tanh7_ps(x: __m128) -> __m128
{
    unsafe {
        let a: __m128       =  _mm_set1_ps(-1.0 / 3.0) ;
        let b: __m128       =  _mm_set1_ps(2.0 / 15.0) ;
        let c: __m128       =  _mm_set1_ps(-17.0 / 315.0) ;
        let one: __m128     =  _mm_set1_ps(1.0) ;
        let xx: __m128      =  _mm_mul_ps(x, x) ;
        let mut y: __m128   =  _mm_add_ps(one, _mm_mul_ps(a, xx)) ;
        let mut x4: __m128  =  _mm_mul_ps(xx, xx) ;

        y  =  _mm_add_ps(y, _mm_mul_ps(b, x4)) ;
        x4 =  _mm_mul_ps(x4, xx)               ;
        y  =  _mm_add_ps(y, _mm_mul_ps(c, x4)) ;

        _mm_mul_ps(y, x)
    }
}


#[cfg(target_arch = "x86_64")] #[inline]
pub fn tanh_fast32(x_in: f32) -> f32 {

    assert!(within_range(0.0, x_in, 1.0));

    use core::arch::x86_64::*;

    let a: f32 = 2.0 / 3.0;
    let x: f32 = x_in.abs();
    let xx: f32 = x * x;

    let mut denom: f32 = 1.0 + x + xx + (a * x * xx);

    unsafe{ _mm_store_ss(&mut denom,
        _mm_rcp_ss(_mm_load_ss(&denom)))};

    match x_in > 0.0 {
        true  => 1.0 * (1.0 - denom),
        false => -1.0 * (1.0 - denom),
    }
}

#[allow(clippy::many_single_char_names)]
#[cfg(target_arch = "x86_64")] #[inline]
pub fn tanh7_ss(x: __m128 ) -> __m128
{
    unsafe {
        let a: __m128      = _mm_set1_ps(-1.0 / 3.0);
        let b: __m128      = _mm_set1_ps(2.0 / 15.0);
        let c: __m128      = _mm_set1_ps(-17.0 / 315.0);
        let one: __m128    = _mm_set1_ps(1.0);
        let xx: __m128     = _mm_mul_ss(x, x);
        let mut y: __m128  = _mm_add_ss(one, _mm_mul_ss(a, xx));
        let mut x4: __m128 = _mm_mul_ss(xx, xx);
        y                  = _mm_add_ss(y, _mm_mul_ss(b, x4));
        x4                 = _mm_mul_ss(x4, xx);
        y                  = _mm_add_ss(y, _mm_mul_ss(c, x4));
        _mm_mul_ss(y, x)
    }
}

#[allow(clippy::many_single_char_names)]
#[cfg(target_arch = "x86_64")] #[inline]
pub fn tanh7_f64(x: f64 ) -> f64
{
    let a: f64 = -1.0 / 3.0;
    let b: f64 = 2.0 / 15.0;
    let c: f64 = -17.0 / 315.0;
    // return tanh(x);
    let xs: f64 = x * x;
    let y: f64 = 1.0 + xs * a + xs * xs * b + xs * xs * xs * c;
    // f64 y = 1 + xs*(a + xs*(b + xs*c));
    // t = xs*c;
    // t += b
    // t *= xs;
    // t += a;
    // t *= xs;
    // t += 1;
    // t *= x;

    y * x
}

#[allow(clippy::many_single_char_names)]
#[cfg(target_arch = "x86_64")]
pub fn tanh7_block<NQ>(xb: *mut f32, nquads: NQ)
    where <NQ as std::convert::TryInto<u32>>::Error: std::fmt::Debug, NQ: TryInto<u32>
{
    let nquads: u32 = nquads.try_into().unwrap();
    unsafe {
        let a: __m128 = _mm_set1_ps(-1.0 / 3.0);
        let b: __m128 = _mm_set1_ps(2.0 / 15.0);
        let c: __m128 = _mm_set1_ps(-17.0 / 315.0);
        let one: __m128 = _mm_set1_ps(1.0);
        let upper_bound: __m128 = _mm_set1_ps(1.1390);
        let lower_bound: __m128 = _mm_set1_ps(-1.1390);

        let mut t: [__m128; 4]  = [z128![]; 4];
        let mut x: [__m128; 4]  = [z128![]; 4];
        let mut xx: [__m128; 4] = [z128![]; 4];

        for i  in (0..nquads).step_by(4)
        {
            x[0] = *(xb as *mut __m128).offset(i as isize);
            x[1] = *(xb as *mut __m128).offset((i + 1) as isize);
            x[2] = *(xb as *mut __m128).offset((i + 2) as isize);
            x[3] = *(xb as *mut __m128).offset((i + 3) as isize);

            x[0] = _mm_max_ps(x[0], lower_bound);
            x[1] = _mm_max_ps(x[1], lower_bound);
            x[2] = _mm_max_ps(x[2], lower_bound);
            x[3] = _mm_max_ps(x[3], lower_bound);
            x[0] = _mm_min_ps(x[0], upper_bound);
            x[1] = _mm_min_ps(x[1], upper_bound);
            x[2] = _mm_min_ps(x[2], upper_bound);
            x[3] = _mm_min_ps(x[3], upper_bound);

            xx[0] = _mm_mul_ps(x[0], x[0]);
            xx[1] = _mm_mul_ps(x[1], x[1]);
            xx[2] = _mm_mul_ps(x[2], x[2]);
            xx[3] = _mm_mul_ps(x[3], x[3]);

            t[0] = _mm_mul_ps(xx[0], c);
            t[1] = _mm_mul_ps(xx[1], c);
            t[2] = _mm_mul_ps(xx[2], c);
            t[3] = _mm_mul_ps(xx[3], c);

            t[0] = _mm_add_ps(t[0], b);
            t[1] = _mm_add_ps(t[1], b);
            t[2] = _mm_add_ps(t[2], b);
            t[3] = _mm_add_ps(t[3], b);

            t[0] = _mm_mul_ps(t[0], xx[0]);
            t[1] = _mm_mul_ps(t[1], xx[1]);
            t[2] = _mm_mul_ps(t[2], xx[2]);
            t[3] = _mm_mul_ps(t[3], xx[3]);

            t[0] = _mm_add_ps(t[0], a);
            t[1] = _mm_add_ps(t[1], a);
            t[2] = _mm_add_ps(t[2], a);
            t[3] = _mm_add_ps(t[3], a);

            t[0] = _mm_mul_ps(t[0], xx[0]);
            t[1] = _mm_mul_ps(t[1], xx[1]);
            t[2] = _mm_mul_ps(t[2], xx[2]);
            t[3] = _mm_mul_ps(t[3], xx[3]);

            t[0] = _mm_add_ps(t[0], one);
            t[1] = _mm_add_ps(t[1], one);
            t[2] = _mm_add_ps(t[2], one);
            t[3] = _mm_add_ps(t[3], one);

            t[0] = _mm_mul_ps(t[0], x[0]);
            t[1] = _mm_mul_ps(t[1], x[1]);
            t[2] = _mm_mul_ps(t[2], x[2]);
            t[3] = _mm_mul_ps(t[3], x[3]);

            *(xb as *mut __m128).offset((i) as isize) = t[0];
            *(xb as *mut __m128).offset((i + 1) as isize) = t[1];
            *(xb as *mut __m128).offset((i + 2) as isize) = t[2];
            *(xb as *mut __m128).offset((i + 3) as isize) = t[3];
        }
    }
}

#[inline] pub fn tanh_fast(x_in: f64) -> f64 {
    let x = x_in.abs();
    let a: f64 = 2.0 / 3.0;
    let xx = x * x;
    let denom: f64 = 1.0 + x + xx + (a * x * xx);
    match x_in > 0.0 {
        true  => 1.0 * (1.0 - 1.0 / denom),
        false => -1.0 * (1.0 - 1.0 / denom),
    }
}

#[inline] pub fn tanh_faster(x_in: f64) -> f64 {
    let a: f64 = -1.0 / 3.0;
    let b: f64 = 2.0 / 15.0;
    let xs: f64 = x_in * x_in;
    let y: f64 = 1.0 + (xs * a) + (xs * xs * b);
    y * x_in
}

pub fn shafted_tanh(x: f64) -> f64
{
    (x.exp() - (-x * 1.2).exp()) /
        ((x).exp() + (-x).exp())
}

