// https://github.com/DLTcollab/sse2neon/blob/master/sse2neon.h

crate::ix!();

use std::arch::aarch64::*;

pub type __m128 = float32x4_t;
pub type __m128i = int32x4_t;

#[macro_export]
macro_rules! vreinterpretq_m128i_s32 {
    ($x: expr) => {
        vreinterpretq_s64_s32(x);
    };
}

#[macro_export]
macro_rules! vreinterpretq_m128_f32 {
    ($x: expr) => {
        $x
    };
}

/**
 * Loads an single - precision, floating - point value into the low word and clears the upper three words.  https://msdn.microsoft.com/en-us/library/548bb9h4%28v=vs.90%29.aspx
 */
#[inline]
pub fn _mm_load_ss(p: &f32) -> float32x4_t {
    return unsafe {
        let result: float32x4_t = vdupq_n_f32(0_f32);
        vsetq_lane_f32(*p, result, 0)
    };
}

// Loads four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/zzd50xxt(v=vs.100).aspx
#[inline]
pub fn _mm_load_ps(p: *const f32) -> float32x4_t {
    return unsafe { vld1q_f32(p) };
}

// Load a single-precision (32-bit) floating-point element from memory into all
// elements of dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load1_ps
#[inline]
pub unsafe fn _mm_load1_ps(p: *const f32) -> float32x4_t {
    return vld1q_dup_f32(p);
}

/**
 * Computes the maximums of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/ff5d607a(v=vs.100).aspx
 */
#[inline]
pub fn _mm_max_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // #if USE_PRECISE_MINMAX_IMPLEMENTATION
    //   return vbslq_f32(vcltq_f32(b,a),a,b);
    // #else
    //   // Faster, but would give inconsitent rendering(e.g. holes, NaN pixels)
    //   return vmaxq_f32(a, b);
    // #endif

    return unsafe { vbslq_f32(vcltq_f32(b, a), a, b) };
}

/**
 * Computes the minima of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/wh13kadz(v=vs.100).aspx
 */
#[inline]
pub fn _mm_min_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // #if USE_PRECISE_MINMAX_IMPLEMENTATION
    //   return vbslq_f32(vcltq_f32(a,b),a,b);
    // #else
    //   // Faster, but would give inconsitent rendering(e.g. holes, NaN pixels)
    //   return vminq_f32(a, b);
    // #endif

    return unsafe { vbslq_f32(vcltq_f32(a, b), a, b) };
}

/**
 * Computes the maximum of the two lower scalar single-precision floating point values of a and b.  https://msdn.microsoft.com/en-us/library/s6db5esz(v=vs.100).aspx
 */
#[inline]
pub fn _mm_max_ss(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    //   float32x4_t value;
    //   float32x4_t result = a;

    let value = _mm_max_ps(a, b);
    return unsafe { vsetq_lane_f32(vgetq_lane_f32(value, 0), a, 0) };
}

/**
 * Computes the minimum of the two lower scalar single-precision floating point values of a and b.  https://msdn.microsoft.com/en-us/library/0a9y7xaa(v=vs.100).aspx
 */
#[inline]
pub fn _mm_min_ss(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let value = _mm_min_ps(a, b);
    return unsafe { vsetq_lane_f32(vgetq_lane_f32(value, 0), a, 0) };
}

/**
 * Stores the lower single - precision, floating - point value. https://msdn.microsoft.com/en-us/library/tzz10fbx(v=vs.100).aspx
 */
#[inline]
pub fn _mm_store_ss(p: *mut f32, a: float32x4_t) {
    unsafe { vst1q_lane_f32(p, a, 0) };
}

/**
 * Stores four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/s3h4ay6y(v=vs.100).aspx
 */
#[inline]
pub fn _mm_store_ps(p: *mut f32, a: float32x4_t) {
    unsafe { vst1q_f32(p, a) };
}

/**
 * Multiplies the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/22kbk6t9(v=vs.100).aspx
 */
#[inline]
pub fn _mm_mul_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    return unsafe { vmulq_f32(a, b) };
}
/**
 * Multiplies the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/22kbk6t9(v=vs.100).aspx
 */
#[inline]
pub fn _mm_mul_ss(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    return unsafe { vmulq_f32(a, b) };
}

// Clears the four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/tk1t2tbz(v=vs.100).aspx
#[inline]
pub unsafe fn _mm_setzero_ps() -> float32x4_t {
    return vdupq_n_f32(0_f32);
}

// Set packed single-precision (32-bit) floating-point elements in dst with the
// supplied values.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set_ps
#[inline]
pub unsafe fn _mm_set_ps(w: f32, z: f32, y: f32, x: f32) -> float32x4_t {
    //   float ALIGN_STRUCT(16) data[4] = {x, y, z, w};

    struct Example {
        x: f32,
        y: f32,
        z: f32,
        w: f32,
    };

    let mew = Example { x, y, z, w };
    return vld1q_f32(&mew as *const Example as *const f32);
}

// Sets the four single-precision, floating-point values to w. https://msdn.microsoft.com/en-us/library/vstudio/2x1se8ha(v=vs.100).aspx
#[inline]
pub fn _mm_set1_ps(_w: f32) -> float32x4_t {
    return unsafe { vdupq_n_f32(_w) };
}

// Sets the four single-precision, floating-point values to w. https://msdn.microsoft.com/en-us/library/vstudio/2x1se8ha(v=vs.100).aspx
#[inline]
pub fn _mm_set_ps1(_w: f32) -> float32x4_t {
    return unsafe { vdupq_n_f32(_w) };
}

// Adds the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/c9848chc(v=vs.100).aspx
#[inline]
pub unsafe fn _mm_add_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    return vaddq_f32(a, b);
}

// adds the scalar single-precision floating point values of a and b.  https://msdn.microsoft.com/en-us/library/be94x2y6(v=vs.100).aspx
#[inline]
pub fn _mm_add_ss(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe {
        let b0 = vgetq_lane_f32(b, 0);
        let mut value = vdupq_n_f32(0_f32);

        //the upper values in the result must be the remnants of <a>.
        value = vsetq_lane_f32(b0, value, 0);
        return vaddq_f32(a, value);
    };
}

// Compute the bitwise AND of packed single-precision (32-bit) floating-point
// elements in a and b, and store the results in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_and_ps
#[inline]
pub unsafe fn _mm_and_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    return vreinterpretq_f32_s32(vandq_s32(
        vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b),
    ));
}

// Subtracts the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/1zad2k61(v=vs.100).aspx
#[inline]
pub fn _mm_sub_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    return unsafe { vsubq_f32(a, b) };
}

#[inline]
pub fn _mm_sub_ss(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    return unsafe { vsubq_f32(a, b) };
}

/*******************************************************/
/* MACRO for shuffle parameter for _mm_shuffle_ps().   */
/* Argument fp3 is a digit[0123] that represents the fp*/
/* from argument "b" of mm_shuffle_ps that will be     */
/* placed in fp3 of result. fp2 is the same for fp2 in */
/* result. fp1 is a digit[0123] that represents the fp */
/* from argument "a" of mm_shuffle_ps that will be     */
/* places in fp1 of result. fp0 is the same for fp0 of */
/* result                                              */
/*******************************************************/

// fn _MN_SHUFFLE(fp3,fp2,fp1,fp0) ( (uint8x16_t){ (((fp3)*4)+0), (((fp3)*4)+1), (((fp3)*4)+2), (((fp3)*4)+3),  (((fp2)*4)+0), (((fp2)*4)+1), (((fp2)*4)+2), (((fp2)*4)+3),  (((fp1)*4)+0), (((fp1)*4)+1), (((fp1)*4)+2), (((fp1)*4)+3),  (((fp0)*4)+0), (((fp0)*4)+1), (((fp0)*4)+2), (((fp0)*4)+3) } )

// fn _MF_SHUFFLE(fp3,fp2,fp1,fp0) { (uint8x16_t){ (((fp3)*4)+0), (((fp3)*4)+1), (((fp3)*4)+2), (((fp3)*4)+3),  (((fp2)*4)+0), (((fp2)*4)+1), (((fp2)*4)+2), (((fp2)*4)+3),  (((fp1)*4)+16+0), (((fp1)*4)+16+1), (((fp1)*4)+16+2), (((fp1)*4)+16+3),  (((fp0)*4)+16+0), (((fp0)*4)+16+1), (((fp0)*4)+16+2), (((fp0)*4)+16+3) } }

// pub fn _MM_SHUFFLE(fp3: u8, fp2: u8, fp1: u8, fp0: u8) -> u8 {
//     ((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0)
// }

#[macro_export]
macro_rules! _MM_SHUFFLE {
    ($fp3: literal, $fp2: literal, $fp1: literal, $fp0: literal) => {
        ((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0)
    };
}

#[macro_export]
macro_rules! vreinterpretq_f32_m128 {
    ($x: expr) => {
        $x
    };
}

// NEON does not support a general purpose permute intrinsic.
// Shuffle single-precision (32-bit) floating-point elements in a using the
// control in imm8, and store the results in dst.
//
// C equivalent:
//   __m128 _mm_shuffle_ps_default(__m128 a, __m128 b,
//                                 __constrange(0, 255) int imm) {
//       __m128 ret;
//       ret[0] = a[imm        & 0x3];   ret[1] = a[(imm >> 2) & 0x3];
//       ret[2] = b[(imm >> 4) & 0x03];  ret[3] = b[(imm >> 6) & 0x03];
//       return ret;
//   }
//
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_ps
// _mm_shuffle_ps_default
// pub fn _mm_shuffle_ps(a: float32x4_t, b: float32x4_t, imm: *const i8) -> float32x4_t {
//     unsafe {
//         vreinterpretq_m128_f32!(vsetq_lane_f32(
//             vgetq_lane_f32(vreinterpretq_f32_m128!(b), ((imm) >> 6) & 0x3),
//             vsetq_lane_f32(
//                 vgetq_lane_f32(vreinterpretq_f32_m128!(b), ((imm) >> 4) & 0x3),
//                 vsetq_lane_f32(
//                     vgetq_lane_f32(vreinterpretq_f32_m128!(a), ((imm) >> 2) & 0x3),
//                     vmovq_n_f32(vgetq_lane_f32(vreinterpretq_f32_m128!(a), (imm) & (0x3))),
//                     1,
//                 ),
//                 2,
//             ),
//             3,
//         ))
//     }
// }

// #[macro_export]
// macro_rules! _mm_shuffle_ps {
//     ($a: ident, $b: ident, $imm: tt) => {
//         unsafe {
//             vreinterpretq_m128_f32(vsetq_lane_f32(
//                 vgetq_lane_f32(vreinterpretq_f32_m128(b), ((imm) >> 6) & 0x3),
//                 vsetq_lane_f32(
//                     vgetq_lane_f32(vreinterpretq_f32_m128(b), ((imm) >> 4) & 0x3),
//                     vsetq_lane_f32(
//                         vgetq_lane_f32(vreinterpretq_f32_m128(a), ((imm) >> 2) & 0x3),
//                         vmovq_n_f32(vgetq_lane_f32(vreinterpretq_f32_m128(a), (imm) & (0x3))),
//                         1,
//                     ),
//                     2,
//                 ),
//                 3,
//             ))
//         }
//     };
// }

// Store 128-bits (composed of 4 packed single-precision (32-bit) floating-point
// elements) from a into memory. mem_addr does not need to be aligned on any
// particular boundary.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_ps
#[inline]
pub unsafe fn _mm_storeu_ps(p: *mut f32, a: float32x4_t) {
    vst1q_f32(p, a);
}

// Load 128-bits (composed of 4 packed single-precision (32-bit) floating-point
// elements) from memory into dst. mem_addr does not need to be aligned on any
// particular boundary.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_ps
#[inline]
pub unsafe fn _mm_loadu_ps(p: *const f32) -> float32x4_t {
    // for neon, alignment doesn't matter, so _mm_load_ps and _mm_loadu_ps are
    // equivalent for neon
    return vld1q_f32(p);
}

// Compute the approximate reciprocal square root of packed single-precision
// (32-bit) floating-point elements in a, and store the results in dst. The
// maximum relative error for this approximation is less than 1.5*2^-12.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rsqrt_ps
#[inline]
pub unsafe fn _mm_rsqrt_ps(i: float32x4_t) -> float32x4_t {
    let mut out: float32x4_t = vrsqrteq_f32(i);

    // Generate masks for detecting whether input has any 0.0f/-0.0f
    // (which becomes positive/negative infinity by IEEE-754 arithmetic rules).
    let pos_inf: uint32x4_t = vdupq_n_u32(0x7F800000);
    let neg_inf: uint32x4_t = vdupq_n_u32(0xFF800000);
    let has_pos_zero: uint32x4_t = vceqq_u32(pos_inf, vreinterpretq_u32_f32(out));
    let has_neg_zero: uint32x4_t = vceqq_u32(neg_inf, vreinterpretq_u32_f32(out));

    out = vmulq_f32(out, vrsqrtsq_f32(vmulq_f32(i, out), out));
    // #if SSE2NEON_PRECISE_SQRT
    //   // Additional Netwon-Raphson iteration for accuracy
    //   out = vmulq_f32(
    //       out, vrsqrtsq_f32(vmulq_f32(vreinterpretq_f32_m128(in), out), out));
    // #endif

    // Set output vector element to infinity/negative-infinity if
    // the corresponding input vector element is 0.0f/-0.0f.
    out = vbslq_f32(has_pos_zero, vreinterpretq_f32_u32(pos_inf), out);
    out = vbslq_f32(has_neg_zero, vreinterpretq_f32_u32(neg_inf), out);

    return out;
}

// Compute the approximate reciprocal square root of the lower single-precision
// (32-bit) floating-point element in a, store the result in the lower element
// of dst, and copy the upper 3 packed elements from a to the upper elements of
// dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rsqrt_ss
#[inline]
pub unsafe fn _mm_rsqrt_ss(i: float32x4_t) -> float32x4_t {
    return vsetq_lane_f32(vgetq_lane_f32(_mm_rsqrt_ps(i), 0), i, 0);
}

// Broadcast 32-bit integer a to all elements of dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set1_epi32
#[inline]
pub unsafe fn _mm_set1_epi32(_i: i32) -> int32x4_t {
    return vdupq_n_s32(_i);
}

// Convert packed single-precision (32-bit) floating-point elements in a to
// packed 32-bit integers with truncation, and store the results in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttps_epi32
#[inline]
pub unsafe fn _mm_cvttps_epi32(a: float32x4_t) -> int32x4_t {
    return vcvtq_s32_f32(a);
}

// Add packed 32-bit integers in a and b, and store the results in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_add_epi32
#[inline]
pub unsafe fn _mm_add_epi32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    return vaddq_s32(a, b);
}

// Cast vector of type __m128i to type __m128. This intrinsic is only used for
// compilation and does not generate any instructions, thus it has zero latency.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_castsi128_ps
#[inline]
pub unsafe fn _mm_castsi128_ps(a: int64x2_t) -> float32x4_t {
    return vreinterpretq_f32_s32(vreinterpretq_s32_s64(a));
}

// #define vreinterpretq_f16_m128(x) vreinterpretq_f16_f32(x)

// Copy the lower single-precision (32-bit) floating-point element of a to dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtss_f32
#[inline]
pub unsafe fn _mm_cvtss_f32(a: float32x4_t) -> f32 {
    return vgetq_lane_f32(a, 0);
}

// Divide packed single-precision (32-bit) floating-point elements in a by
// packed elements in b, and store the results in dst.
// Due to ARMv7-A NEON's lack of a precise division intrinsic, we implement
// division by multiplying a by b's reciprocal before using the Newton-Raphson
// method to approximate the results.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_div_ps
#[inline]
pub unsafe fn _mm_div_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // #if defined(__aarch64__) || defined(_M_ARM64)
    //   return vreinterpretq_m128_f32(
    //       vdivq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
    // #else
    //   float32x4_t recip = vrecpeq_f32(vreinterpretq_f32_m128(b));
    //   recip = vmulq_f32(recip, vrecpsq_f32(recip, vreinterpretq_f32_m128(b)));
    //   // Additional Netwon-Raphson iteration for accuracy
    //   recip = vmulq_f32(recip, vrecpsq_f32(recip, vreinterpretq_f32_m128(b)));
    //   return vreinterpretq_m128_f32(vmulq_f32(vreinterpretq_f32_m128(a), recip));
    // #endif

    return vdivq_f32(a, b);
}

// Compute the bitwise AND of 128 bits (representing integer data) in a and b,
// and store the result in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_and_si128
#[inline]
pub unsafe fn _mm_and_si128(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    return vandq_s32(a, b);
}

// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and
// store the results in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sub_epi32
#[inline]
pub unsafe fn _mm_sub_epi32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    return vsubq_s32(a, b);
}

// Convert packed signed 32-bit integers in a to packed single-precision
// (32-bit) floating-point elements, and store the results in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepi32_ps
#[inline]
pub unsafe fn _mm_cvtepi32_ps(a: int32x4_t) -> float32x4_t {
    return vcvtq_f32_s32(a);
}

// Compute the approximate reciprocal of packed single-precision (32-bit)
// floating-point elements in a, and store the results in dst. The maximum
// relative error for this approximation is less than 1.5*2^-12.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rcp_ps
#[inline]
pub unsafe fn _mm_rcp_ps(i: float32x4_t) -> float32x4_t {
    let mut recip: float32x4_t = vrecpeq_f32(i);
    recip = vmulq_f32(recip, vrecpsq_f32(recip, i));
    // TODO: SSE2NEON_PRECISE_DIV
    // #if SSE2NEON_PRECISE_DIV
    //   // Additional Netwon-Raphson iteration for accuracy
    //   recip = vmulq_f32(recip, vrecpsq_f32(recip, vreinterpretq_f32_m128(in)));
    // #endif
    return recip;
}

// Convert the lower single-precision (32-bit) floating-point element in a to a
// 32-bit integer, and store the result in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvt_ss2si
#[inline]
pub unsafe fn _mm_cvt_ss2si(a: float32x4_t) -> i32 {
    // #if (defined(__aarch64__) || defined(_M_ARM64)) || \
    //     defined(__ARM_FEATURE_DIRECTED_ROUNDING)
    //   return vgetq_lane_s32(vcvtnq_s32_f32(vrndiq_f32(vreinterpretq_f32_m128(a))),
    //                         0);
    // #else
    //   float32_t data = vgetq_lane_f32(
    //       vreinterpretq_f32_m128(_mm_round_ps(a, _MM_FROUND_CUR_DIRECTION)), 0);
    //   return (int32_t)data;
    // #endif
    return vgetq_lane_s32(vcvtnq_s32_f32(vrndiq_f32(a)), 0);
}
