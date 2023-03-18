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
