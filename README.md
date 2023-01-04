# mlir-rustc


```console
> export LLVM_BUILD_DIR=

> export LLVM=$LLVM_BUILD_DIR/lib/cmake/llvm
> export MLIR=$LLVM_BUILD_DIR/lib/cmake/mlir
> cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR .
> make
```

# Requirements

* trunk LLVM/MLIR
* cmake

```console
>  brew install llvm
```


# TODO

- [ ] keywords
- [ ] UTF-8
- [ ] Remarks
- [ ] Mir to LLVM
- [ ] async and await
- [ ] Closures (nested regions)
- [ ] visibility
- [ ] visibility checks in inliner
- [ ] desugaring: AST -> HIR -> MIR
- [x] location in Lexer
- [ ] convert ModuleBuilder to CrateBuilder for code generation

# References

* https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/
* https://github.com/zero9178/Pylir
* https://github.com/intel/llvm/tree/sycl-mlir/mlir-sycl
* https://mlir.llvm.org/docs/Dialects/ArithOps/

## Async

* https://mlir.llvm.org/docs/Dialects/AsyncDialect/
* https://discourse.llvm.org/t/rfc-new-dialect-for-modelling-asynchronous-execution-at-a-higher-level/1345/11

## Rust

* https://doc.rust-lang.org/reference/expressions/await-expr.html
* https://doc.rust-lang.org/reference/expressions/operator-expr.html#borrow-operators
* https://github.com/LightningCreations/lccc
* https://github.com/thepowersgang/mrustc
* https://rustc-dev-guide.rust-lang.org/about-this-guide.html

## MIR

* https://blog.rust-lang.org/2016/04/19/MIR.html
* https://rustc-dev-guide.rust-lang.org/mir/index.html


```bibtex
@inproceedings{lattner2021mlir,
  title={Mlir: Scaling compiler infrastructure for domain specific computation},
  author={Lattner, Chris and Amini, Mehdi and Bondhugula, Uday and Cohen, Albert and Davis, Andy and Pienaar, Jacques and Riddle, River and Shpeisman, Tatiana and Vasilache, Nicolas and Zinenko, Oleksandr},
  booktitle={2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
  pages={2--14},
  year={2021},
  organization={IEEE}
}
````



# Ideas

* https://icl.utk.edu/papi/

* PGO + Instruction counters: where to optimize and not how to optimize



* https://reviews.llvm.org/D137956
