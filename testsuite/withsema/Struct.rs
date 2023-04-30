struct SmallVector<T, const N: usize> {
    data: [T; N],
}

struct StaticVector<T = i32> {
    data: [T; 4],
}

fn foo() {
    let x: SmallVector<i32, 8>;

    let y: StaticVector<i32>;
}

fn main() {
    foo()
}
