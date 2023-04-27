struct SmallVector<T, const N: usize> {
    data: [T; N],
}

fn foo() {
    let x: SmallVector<i32, 8>;
}

fn main() {
    foo()
}
