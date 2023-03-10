#![allow(unused)]
fn main() {
    // The function is only included in the build when compiling for macOS
    #[cfg(target_os = "macos")]
    fn macos_only() {
        // ...
    }

    // This function is only included when either foo or bar is defined
    #[cfg(any(foo, bar))]
    fn needs_foo_or_bar() {
        // ...
    }

    // This function is only included when compiling for a unixish OS with a 32-bit
    // architecture
    #[cfg(all(unix, target_pointer_width = "32"))]
    fn on_32bit_unix() {
        // ...
    }

    // This function is only included when foo is not defined
    #[cfg(not(foo))]
    fn needs_not_foo() {
        // ...
    }

    // This function is only included when the panic strategy is set to unwind
    #[cfg(panic = "unwind")]
    fn when_unwinding() {
        // ...
    }
}
