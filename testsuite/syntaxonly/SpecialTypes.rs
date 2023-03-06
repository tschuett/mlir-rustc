struct Foo {}

fn foo() {
    let x: Box<Foo> = 1;
    let x: Rc<Foo> = 1;
    let x: Arc<Foo> = 1;
}
