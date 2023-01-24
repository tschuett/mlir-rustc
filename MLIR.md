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


```MLIR
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: ui64, %arg1: ui64):
    %0 = "memref.alloc"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<1xi64>
    "func.return"(%0) : (memref<1xi64>) -> ()
  }) {function_type = (ui64, ui64) -> ui64, sym_name = "add"} : () -> ()
}) : () -> ()
```
