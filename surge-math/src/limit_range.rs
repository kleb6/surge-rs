crate::ix!();

pub trait LimitRange: Sized {
    fn limit_range(self, low: Self, hi: Self) -> Self;

    /**
    | provides a possibility for taking a
    | non-simd path. if we don't customize it,
    | the ordinary simd code will be called if it
    | is used in "limit_range"
    |
    */
    #[inline]
    fn limit_range_nosimd(self, low: Self, hi: Self) -> Self {
        self.limit_range(low, hi)
    }
}

pub fn limit_range<T: LimitRange>(x: T, low: T, high: T) -> T {
    x.limit_range(low, high)
}

impl LimitRange for f32 {
    // #[cfg(target_arch = "x86_64")]
    #[inline]
    fn limit_range(self, low: Self, high: Self) -> Self {
        let mut result: f32 = 0.0;

        unsafe {
            let _s = _mm_load_ss(&self);
            let _low = _mm_load_ss(&low);
            let _high = _mm_load_ss(&high);
            let limited = _mm_min_ss(_mm_max_ss(_s, _low), _high);

            _mm_store_ss(&mut result, limited)
        };

        result
    }
    fn limit_range_nosimd(self, min: f32, max: f32) -> f32 {
        std::cmp::max(std::cmp::min(FloatOrd(self), FloatOrd(max)), FloatOrd(min)).0
    }
}

impl LimitRange for i32 {
    fn limit_range(self, l: i32, h: i32) -> i32 {
        std::cmp::max(std::cmp::min(self, h), l)
    }
}

impl LimitRange for u32 {
    fn limit_range(self, l: u32, h: u32) -> u32 {
        std::cmp::max(std::cmp::min(self, h), l)
    }
}

impl LimitRange for f64 {
    fn limit_range(self, min: f64, max: f64) -> f64 {
        std::cmp::max(std::cmp::min(FloatOrd(self), FloatOrd(max)), FloatOrd(min)).0
    }
}
