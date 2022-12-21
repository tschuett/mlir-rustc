#include "CrateBuilder.h"
#include "mlir/IR/MLIRContext.h"

#include "Mir/MirDialect.h"

namespace rust_compiler {

void CrateBuilder::build(AST *) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::mir::Mir::MirDialect>();
}

} // namespace rust_compiler
