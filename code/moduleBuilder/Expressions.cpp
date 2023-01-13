#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/Expression.h"
#include "AST/OperatorExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

using namespace llvm;
using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitExpression(std::shared_ptr<Expression> expr) {
  ExpressionKind kind = expr->getExpressionKind();

  switch (kind) {
  case ExpressionKind::ExpressionWithBlock: {
    return emitExpressionWithBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithBlock>(expr));
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    return emitExpressionWithoutBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithoutBlock>(expr));
  }
  }
}


void ModuleBuilder::emitReturnExpression(
    std::shared_ptr<ast::ReturnExpression> ret) {
  // mlir::Value reti = emitExpression(ret->getExpression());

  // builder.create<mlir::func::ReturnOp>(getLocation(ret->getLocation()));

  if (ret->getExpression()) {
    std::shared_ptr<ast::Expression> expr = ret->getExpression();
    mlir::Value mlirExpr = emitExpression(ret->getExpression());
    builder.create<mlir::func::ReturnOp>(getLocation(ret->getLocation()),
                                         ArrayRef(mlirExpr));
  } else {
    builder.create<mlir::func::ReturnOp>(getLocation(ret->getLocation()));
  }
}

mlir::Value ModuleBuilder::emitOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> opr) {

  switch (opr->getKind()) {
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    return emitArithmeticOrLogicalExpression(
        static_pointer_cast<ArithmeticOrLogicalExpression>(opr));
  }
  default: {
    assert(false);
  }
  }

  return nullptr;
}

} // namespace rust_compiler
