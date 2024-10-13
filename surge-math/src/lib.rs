#![allow(soft_unstable)]
#![allow(internal_features)]
#![allow(unused_imports)]
#![feature(test)]
#![feature(core_intrinsics)]
// #[cfg(target_arch = "x86_64")]
// #![feature(stdarch_x86_mm_shuffle)]
#![feature(trait_alias)]

#[macro_use]
mod imports;
use imports::*;
#[macro_use]
pub mod m128;

x! {align}
x! {error}
x! {fill}
x! {allpass}
x! {basic_ops}
x! {clamp}
x! {clear}
x! {convert}
x! {copy}
x! {correlated_noise}
x! {encode}
x! {get_sign}
x! {absmax}
x! {access}
x! {mul}
x! {sub}
x! {sum}
x! {add}
x! {accum}
x! {fast}
x! {flops}
x! {hardclip}
x! {interp}
x! {limit_range}
x! {ord}
x! {padding}
x! {pan}
x! {saturate}
x! {scan}
x! {scratch}
x! {simd}
x! {sinc}
x! {sine}
x! {softclip}
x! {square}
x! {tanh}
x! {trixpan}
x! {utility}
x! {value_cast}
x! {white_noise}
x! {window}
