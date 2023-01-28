include "mlir/IR/OpBase.td"

class Mir_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::rust_compiler::Mir";
}