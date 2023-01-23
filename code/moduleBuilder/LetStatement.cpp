#include "ModuleBuilder/ModuleBuilder.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace rust_compiler {

void ModuleBuilder::buildLetStatement(
    std::shared_ptr<ast::LetStatement> letStmt) {
  // FIXME
  assert(false);

  for (std::string &lit : letStmt->getPattern()->getLiterals()) {
    mlir::Value alloca = builder.create<mlir::memref::AllocOp>(
        getLocation(letStmt->getLocation()), TypeRange(builder.getI1Type()));
    symbolTable.insert(lit, {alloca, lit});
  }
}

} // namespace rust_compiler
