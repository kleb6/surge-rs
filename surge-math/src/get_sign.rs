use std::arch::asm;

crate::ix!();

/// Returns the sign of an i32 integer as -1 or 1.
///
/// On x86_64 target architecture, this function uses
/// a simple match expression.
///
/// On other architectures, the function uses inline
/// assembly.
///
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn sign(x: i32) -> i32 {
    match x < 0 {
        true => -1,
        false => 1,
    }
}

/// Returns the sign of an i32 integer as -1 or 1 using
/// inline assembly.
///
/// This function uses inline assembly to determine the sign
/// of the input value x.
///
/// Note that inline assembly is only supported on the
/// nightly Rust.
///
#[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
pub fn sign(x: i32) -> i32 {
    let result: i32;
    unsafe {
        asm!(
            "mov eax, 1",
            "mov edx, -1",
            "test {input}, {input}",
            "cmovs eax, edx",
            input = in(reg) x,
            out("eax") result,
        );
    }
    result
}
