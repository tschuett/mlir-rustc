struct SmallVector<const N: usize> {
    data: [i32; N],
}

fn foo() {
    let x: SmallVector<8>;
}
