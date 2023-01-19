#include "ModuleBuilder/ModuleBuilder.h"

using namespace mlir;
using namespace llvm;
using namespace rust_compiler::adt;

namespace rust_compiler {

std::optional<mlir::Value>
ModuleBuilder::emitBlockExpression(std::shared_ptr<ast::BlockExpression> blk) {
  llvm::outs() << "block count: " << symbolTable.contains("right") << "\n";

  ScopedHashTableScope<llvm::StringRef,
                       std::pair<mlir::Value, ast::VariableDeclaration *>>
      varScope(symbolTable);

  llvm::outs() << "block count: " << symbolTable.contains("right") << "\n";

  std::optional<mlir::Value> result = std::nullopt;

  llvm::outs() << "emitBlockExpression"
               << "\n";

  // new variable scope?
  result = emitStatements(blk->getExpressions());

  return result;
}

} // namespace rust_compiler
