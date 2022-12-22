#pragma once

namespace rust_compiler {

int processMLIR(mlir::MLIRContext &context,
                mlir::OwningOpRef<mlir::ModuleOp> &module);

}
