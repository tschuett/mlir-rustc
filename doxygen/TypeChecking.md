* Type Checking

Note that the types of the integer literals in the two functions
differ. The root cause are the different return types of the two
functions.

```c
fn foo() -> i32 {
  return 1 + 2;
}

fn bar() -> u128 {
  return 1 + 2;
}
```

The rust compiler performs type checking in the [rustc_hir_analysis](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html) crate.


[TyCtx](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html)
[ty](https://rustc-dev-guide.rust-lang.org/ty.html)

# Generics

Note that the type of T in the data array is unknown until the
instantiation of SmallVector.

```
struct SmallVector<T, const N: usize> {
    data: [T; N],
}

fn foo() {
    let x: SmallVector<i32, 8>;
}
```

# Unification

# Coercion

[Type coercions](https://doc.rust-lang.org/reference/type-coercions.html)

For example, &mut 42 is coerced to have type &i8 in the following:
```c
fn bar(_: &i8) { }

fn main() {
    bar(&mut 42);
}
```

# Cast

```c
f as f64
```
