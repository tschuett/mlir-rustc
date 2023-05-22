fn main() {
    let a: *mut i32;
    unsafe {
        let b = [13u8, 17u8];
        let a = &b[0] as *const u8;
    }
}
