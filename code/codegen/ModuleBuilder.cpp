#include "ModuleBuilder.h"

#include "Mir/MirDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

namespace rust_compiler {

void ModuleBuilder::build(std::shared_ptr<ast::Module> mod) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::mir::Mir::MirDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
}

} // namespace rust_compiler
