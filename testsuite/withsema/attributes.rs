// General metadata applied to the enclosing module or crate.
#![crate_type = "lib"]

// A function marked as a unit test
#[test]
fn test_foo() {
    /* ... */
}

// A conditionally-compiled module
#[cfg(target_os = "linux")]
mod bar {
    /* ... */
}

#[derive(Serialize)]
struct Foo {
    #[doc = include_str!("x.md")]
    x: u32,
}

// A lint attribute used to suppress a warning/error
#[allow(non_camel_case_types)]
type int8_t = i8;

// Inner attribute applies to the entire function.
fn some_unused_variables() {
    #![allow(unused_variables)]

    let x = ();
    let y = ();
    let z = ();
}


#![allow(unused)]
fn main() {
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn foo_avx2() {}
}

#[inline(never)]
fn inliner() {
}


#[must_use]
struct MustUse {
    // some fields
}

#![allow(unused)]
fn main() {
// Default representation, alignment lowered to 2.
#[repr(packed(2))]
struct PackedStruct {
    first: i16,
    second: i8,
    third: i32
}

// C representation, alignment raised to 8
#[repr(C, align(8))]
struct AlignedStruct {
    first: i16,
    second: i8,
    third: i32
}
}

