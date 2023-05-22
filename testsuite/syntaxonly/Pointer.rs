fn main() {
    let a: *mut i32;
    unsafe {
        let b = [13, 17];
        let a = &b[0] as *const u8;
    }
}
