pub enum Option<T> {
    None,
    Some(T),
}

fn main() {
    let a = if let Option::Some(1) = x {
        1
    };
}
