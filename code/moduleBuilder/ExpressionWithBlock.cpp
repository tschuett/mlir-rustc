#include "AST/BlockExpression.h"
#include "AST/Expression.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <memory>

namespace rust_compiler {

using namespace rust_compiler::ast;

mlir::Value ModuleBuilder::emitExpressionWithBlock(
    std::shared_ptr<ExpressionWithBlock> expr) {

  llvm::outs() << "emitExpressionWithBlock"
               << "\n";

  switch (expr->getKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    std::shared_ptr<ast::BlockExpression> blk =
        std::static_pointer_cast<ast::BlockExpression>(expr);
    std::optional<mlir::Value> result = emitBlockExpression(blk);
    if (result)
      return *result;
    return nullptr;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    break;
  }
  }
  llvm::outs() << "emitExpressionWithBlock: failed"
               << "\n";

  // FIXME
  return nullptr;
}

} // namespace rust_compiler
