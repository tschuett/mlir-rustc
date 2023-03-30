* Name Resolution

[rustc](https://rustc-dev-guide.rust-lang.org/name-resolution.html)

Is `a` in the return expression, the function parameter, the variable introduced by the let statement, or the global constant?
```c
const a: i32 = 3;

fn foo(a: i32) {
  let a = 5;
  return a + 2;
}
```

[rustc_resolve](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/index.html)

# Resolver

[Resolver](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html)

# Scope

# Rib

Variables, types, and labels are in different namespaces.

[Rib](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.Rib.html)

[RibKind](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/enum.RibKind.html)

# Mappings
