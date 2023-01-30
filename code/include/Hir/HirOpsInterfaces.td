include "mlir/IR/OpBase.td"

class Hir_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::rust_compiler::Hir";
}