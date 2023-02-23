fn foo(a: &[u32]) {
    match a {
        [first, ..] => {}
        [.., last] => {}
        _ => {}
    }
}

fn foo() {
    let [] = [0; 0];
}
