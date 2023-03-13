fn add(x: i32, y: i32) -> i32 {
    x + y;
}

fn main() {
    let mut x = add(5, 7);

    type Binop = fn(i32, i32) -> i32;
    let bo: Binop = add;
    x = bo(5, 7);
}
