#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace rust_compiler {

int processMLIR(mlir::MLIRContext &context,
                mlir::OwningOpRef<mlir::ModuleOp> &module);

}

