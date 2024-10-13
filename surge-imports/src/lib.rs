#![allow(unused_imports)]

/*
#[macro_export] macro_rules! xt {
    ($x:ident) => {
        #[cfg(test)] mod $x;
    }
}
*/

#[cfg(target_arch = "x86_64")]
pub fn assert_m128_eq(a: __m128, b: __m128, epsilon: f32) {

    let a_arr: [f32; 4] = unsafe { std::mem::transmute(a) };
    let b_arr: [f32; 4] = unsafe { std::mem::transmute(b) };

    for i in 0..4 {
        assert!(
            (a_arr[i] - b_arr[i]).abs() <= epsilon,
            "Vectors not equal at index {}: {:?} != {:?}",
            i,
            a_arr,
            b_arr
        );
    }
}

#[cfg(target_arch = "x86_64")]
pub use core::arch::x86_64::*;

pub use disable_macro::disable;
pub use tracing::{self,debug,info,warn,error};
pub use tracing_subscriber;
pub use traced_test::*;
pub use tracing_setup::*;
pub use rustfft::{FftPlanner, num_complex::Complex};
pub use export_magic::*;
pub use error_tree::error_tree;
pub use std::ptr::NonNull;
pub use std::alloc::{Layout,alloc,dealloc};
pub use std::mem::MaybeUninit;
pub use std::mem;
pub use ::approx_eq::*;
pub use approx::{self,assert_abs_diff_eq};
pub use atomic_float::{AtomicF64,AtomicF32};
pub use auto_impl::*;
pub use bitflags::*;
pub use byteorder::{ByteOrder,BigEndian,LittleEndian};
pub use core::convert::*;
pub use core::ops::*;
pub use coreaudio::audio_unit::*;
pub use coreaudio::Error;
pub use derivative::*;
pub use derive_more::*;
pub use dft::*;
pub use dft;
pub use std::cmp::Ordering;
pub use downcast_rs::*;
pub use enhanced_enum::*;
pub use enum_dispatch::*;
pub use float_ord::*;
pub use delegate::*;
pub use getset::*;
pub use indoc::*;
pub use lazy_static::*;
pub use libc;
pub use lockfree::map::Map as LFMap;
pub use ndarray;
pub use ndarray::{Axis,array};
pub use num::*;
pub use num::complex::*;
pub use num_traits::*;
pub use num_traits::pow::*;
pub use num_traits;
pub use paste::*;
pub use proc_macro2::*;
pub use rand::{self,distributions,thread_rng,Rng};
//pub use coreaudio::audio_unit::render_callback::*;
//pub use half::*;
//pub use mopa::*;
//pub use prometheus::core::*;
//pub use rand;
pub use regex::Regex;
pub use std::iter::Rev;
pub use std::sync::{Arc,Mutex};
pub use std::sync::atomic::{AtomicI32};

pub use vst::api::{Events,Supported};
pub use vst::host::{Host,PluginInstance,PluginLoader};
pub use vst::buffer::AudioBuffer;
pub use vst::plugin::{Plugin,Category,Info,CanDo};
pub use coreaudio::audio_unit::render_callback::data::NonInterleaved;
//pub use vst::event::*;
pub use serde::*;
pub use std::any::Any;
pub use std::cell::*;
pub use std::cmp::*;
pub use std::collections::*;
pub use std::convert::*;
pub use std::f32::consts::PI as PI_32;
pub use std::f64::consts::PI as PI;
pub use std::ffi::*;
pub use std::ffi;
pub use std::fmt;
pub use std::fmt::Debug;

pub use std::fs::{
    File,
    OpenOptions,
};

pub use std::io::{
    Write,
    BufReader,
    BufRead,
};

pub use std::marker::*;
pub use std::ops::*;
pub use std::path::*;
pub use std::pin::*;
pub use std::rc::*;
pub use std::sync::atomic;
pub use std::thread;
pub use std::time::*;
pub use std::time;
pub use uuid::Uuid;
