fn foo() {
    local_var;
    globals::STATIC_VAR;
    unsafe { globals::STATIC_MUT_VAR };
    let some_constructor = Some::<i32>;
    let push_integer = Vec::<i32>::push;
    let slice_reverse = <[i32]>::reverse;
}

fn bar() {
    (0..10).collect::<Vec<_>>();
    Vec::<u8>::with_capacity(1024);
}
