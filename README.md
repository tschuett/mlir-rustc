# mlir-rustc


```console
> export LLVM_BUILD_DIR=

> export LLVM=$LLVM_BUILD_DIR/lib/cmake/llvm
> export MLIR=$LLVM_BUILD_DIR/lib/cmake/mlir
> cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR .
> make
> ctest
```

# Requirements

* trunk LLVM/MLIR
* cmake
* Google Test
* icu
* doxygen


# TODO

- [x] keywords
- [x] Rewriter for borrow ops
- [ ] UTF-8
- [ ] Remarks
- [ ] Mir to LLVM
- [ ] async and await
- [ ] Closures (nested regions)
- [ ] visibility
- [ ] visibility checks in inliner
- [x] location in Lexer
- [ ] unsafe checker
- [ ] cover all tokens
- [ ] types
- [ ] patterns
- [ ] error progation
- [ ] type uniquing or equivalence?
- [ ] iterators
- [ ] .pcm
- [ ] outer attributes on expressions
- [ ] outer attributes on item
- [ ] macros
- [ ] precedence
- [ ] rustc prefers One-symbol tokens. Solution for Axum?
- [ ] drops
- [ ] copy

# References

* https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/
* https://github.com/zero9178/Pylir
* https://github.com/intel/llvm/tree/sycl-mlir/mlir-sycl
* https://mlir.llvm.org/docs/Dialects/ArithOps/
* https://discourse.llvm.org/t/rfc-a-dataflow-analysis-framework/63340
* https://llvm.org/devmtg/2019-10/slides/Doerfert-Attributor.pdf
* https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/Analysis/DFX/Solver.h
* https://llvm.org/devmtg/2020-09/slides/The_Present_and_Future_of_Interprocedural_Optimization_in_LLVM.pdf
* https://llvm.org/devmtg/2020-09/slides/A_Deep_Dive_into_Interprocedural_Optimization.pdf
* https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/en/latest/
* https://discourse.llvm.org/t/representing-anonymous-functions-lambdas/4962/3

## Async

* https://mlir.llvm.org/docs/Dialects/AsyncDialect/
* https://discourse.llvm.org/t/rfc-new-dialect-for-modelling-asynchronous-execution-at-a-higher-level/1345/11
* https://discourse.llvm.org/t/value-range-analysis-of-source-code-variables/62853/5
* https://discourse.llvm.org/t/upstreaming-from-our-mlir-python-compiler-project/64931

## Rust

* https://doc.rust-lang.org/reference/expressions/await-expr.html
* https://doc.rust-lang.org/reference/expressions/operator-expr.html#borrow-operators
* https://github.com/LightningCreations/lccc
* https://github.com/thepowersgang/mrustc
* https://rustc-dev-guide.rust-lang.org/about-this-guide.html
* https://spec.ferrocene.dev/index.html
* https://github.com/Rust-GCC/gccrs

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

* ThinLTO at MLIR

* https://reviews.llvm.org/D137956

* attributer on Mir

* https://discourse.llvm.org/t/rfc-globalisel-replace-the-current-globalisel-matcher-with-a-bottom-up-matcher/67530

* https://discourse.llvm.org/t/thinlto-import-functions-to-enable-function-specialization-in-thinlto/58627

```console
> tokei
```


# Dialects

* Mir
* arith
* func
* async
* memref
* controlflow




https://reviews.llvm.org/D142244

https://reviews.llvm.org/D141820

https://reviews.llvm.org/D142897

* Distributed ThinLTO for MachO
https://reviews.llvm.org/D138451

* CycleInfo
https://reviews.llvm.org/D112696

* ThinLTO for MachO
https://reviews.llvm.org/D90663

* __builtin_assume_separate_storage
https://reviews.llvm.org/D136515

* hlfir.forall
https://reviews.llvm.org/D149734

* assignment mask operations
https://reviews.llvm.org/D149754


* A new code layout algorithm for function reordering
https://reviews.llvm.org/D152834

* [FLang] Add support for Rpass flag
https://reviews.llvm.org/D156320



[GlobalISel] Introduce global variant of regbankselect
* https://reviews.llvm.org/D90304
