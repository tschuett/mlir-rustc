trait Fn {}

mod math {
    pub type Complex = (f64, f64);

    pub fn sin(f: f64) -> f64 {
        1.0
    }
    fn cos(f: f64) -> f64 {
        2.0
    }
    fn tan(f: f64) -> f64 {
        3.0
    }
}

mod generic {
    pub trait Trait {}

    struct Trades {}

    impl Trait for Trades {}

    impl Trades {
        pub fn new() -> Self {
            return Trades {};
        }
    }

    fn foo(arg: impl Trait) {}

    fn bar() -> impl Trait {
        return Trades::new();
    }

    fn foo2<T: Trait>(arg: T) {}

    fn returns_closure() -> impl Fn(i32) -> i32 {
        |x| x + 1
    }
}

fn main() {
    math::sin(5.0);

    let complex: math::Complex;
    ()
}
