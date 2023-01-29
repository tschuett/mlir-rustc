#ifndef HMirOps
#define HMirOps


include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "HMirDialect.td"
include "HMirTypes.td"
include "HMirAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "HMirInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Ops
class HMir_Op<string mnemonic, list<Trait> traits = []> :
    Op<HMir_Dialect, mnemonic, traits>;



#endif