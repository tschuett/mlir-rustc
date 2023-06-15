trait A<T> {
    type Output;

    fn test(self, a: &T) -> &Self::Output;
}

struct Foo<T> {
    start: T,
    end: T,
}

impl<T> A for Foo<usize> {
    type Output = T;

    fn test(self, a: &T) -> &Self::Output {
        a
    }
}
