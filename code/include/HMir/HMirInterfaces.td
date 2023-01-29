include "mlir/IR/OpBase.td"

class HMir_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::rust_compiler::HMir";
}