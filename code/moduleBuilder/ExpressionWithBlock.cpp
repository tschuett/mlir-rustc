#include "AST/BlockExpression.h"
#include "AST/Expression.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/LetStatement.h"
#include "AST/LoopExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <memory>

namespace rust_compiler {

using namespace rust_compiler::ast;

mlir::Value ModuleBuilder::emitExpressionWithBlock(
    std::shared_ptr<ExpressionWithBlock> expr) {

  llvm::outs() << "emitExpressionWithBlock: "
               << uint32_t(expr->getWithBlockKind()) << "\n";

  switch (expr->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    llvm::outs() << "emitExpressionWithBlock: block"
                 << "\n";
    std::shared_ptr<ast::BlockExpression> blk =
        std::static_pointer_cast<ast::BlockExpression>(expr);
    std::optional<mlir::Value> result = emitBlockExpression(blk);
    if (result)
      return *result;
    return nullptr;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    llvm::outs() << "emitExpressionWithBlock: unsafe"
                 << "\n";
    exit(EXIT_FAILURE);
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {

    llvm::outs() << "emitExpressionWithBlock: loop"
                 << "\n";
    std::shared_ptr<ast::LoopExpression> loop =
        std::static_pointer_cast<ast::LoopExpression>(expr);

    return emitLoopExpression(loop);

    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    llvm::outs() << "emitExpressionWithBlock: if"
                 << "\n";
    std::shared_ptr<ast::IfExpression> ifExpr =
        std::static_pointer_cast<ast::IfExpression>(expr);
    return emitIfExpression(ifExpr);
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    llvm::outs() << "emitExpressionWithBlock: iflet"
                 << "\n";

    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    llvm::outs() << "emitExpressionWithBlock: match"
                 << "\n";
    break;
  }
  case ExpressionWithBlockKind::Unknown: {
    llvm::outs() << "emitExpressionWithBlock: unknown UPS!!!!"
                 << "\n";
    break;
  }
  }
  llvm::outs() << "emitExpressionWithBlock: failed"
               << "\n";

  // FIXME
  return nullptr;
}

} // namespace rust_compiler
