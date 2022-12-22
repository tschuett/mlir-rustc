#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace rust_compiler {

int dumpLLVMIR(mlir::ModuleOp module);

}
