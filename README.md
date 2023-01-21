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
- [ ] desugaring: AST -> HIR -> MIR
- [x] location in Lexer
- [ ] convert ModuleBuilder to CrateBuilder for code generation
- [ ] unsafe checker
- [ ] cover all tokens
- [ ] types
- [ ] patterns
- [ ] error progation

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
* https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/en/latest/README.html
## Async

* https://mlir.llvm.org/docs/Dialects/AsyncDialect/
* https://discourse.llvm.org/t/rfc-new-dialect-for-modelling-asynchronous-execution-at-a-higher-level/1345/11
* https://discourse.llvm.org/t/value-range-analysis-of-source-code-variables/62853/5

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

* ThinLTO at MLIR

* https://reviews.llvm.org/D137956

* attributer on Mir

* https://discourse.llvm.org/t/rfc-globalisel-replace-the-current-globalisel-matcher-with-a-bottom-up-matcher/67530



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



```MLIR
module {
  func.func @add(%arg0: ui64) -> ui64 attributes {"function type" = "async", visibility = "pub"} {
    %0 = "mir.addi"(%arg0, %arg0) : (ui64, ui64) -> ui64
    return %0 : ui64
  }
}
```

```MLIR
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: ui64):  // 2 preds: ^bb1, ^bb2
    "func.return"(%arg0) : (ui64) -> ()
  ^bb1:  // pred: ^bb3
    %0 = "mir.constant"() {value = 5 : i64} : () -> i64
    "mir.br"(%0)[^bb0] : (i64) -> ()
  ^bb2:  // pred: ^bb3
    %1 = "mir.constant"() {value = 6 : i64} : () -> i64
    "mir.br"(%1)[^bb0] : (i64) -> ()
  ^bb3(%2: ui64):  // no predecessors
    %3 = "mir.constant"() {value = true} : () -> i1
    "mir.cond_br"(%3)[^bb1, ^bb2] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  }) {"function type" = "async", function_type = (ui64) -> ui64, sym_name = "add", visibility = "pub"} : () -> ()
```


https://reviews.llvm.org/D142244

https://reviews.llvm.org/D141820
