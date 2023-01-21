#include "AST/IfExpression.h"

#include "AST/Expression.h"
#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "mlir/IR/Value.h"

#include <memory>
#include <optional>
#include <vector>

using namespace rust_compiler::ast;

namespace rust_compiler {

static bool isIfExpression(std::shared_ptr<ast::Expression> expr) {
  if (expr->getExpressionKind() == ast::ExpressionKind::ExpressionWithBlock) {
    std::shared_ptr<ast::ExpressionWithBlock> withBlock =
        std::static_pointer_cast<ast::ExpressionWithBlock>(expr);
    if (withBlock->getWithBlockKind() ==
        ast::ExpressionWithBlockKind::IfExpression) {
      return true;
    }
  }
  return false;
}

static uint32_t getIfBlocks(std::shared_ptr<ast::Expression> expr) {
  if (isIfExpression(expr)) {
    std::shared_ptr<ast::IfExpression> ifExpr =
        std::static_pointer_cast<ast::IfExpression>(expr);
    if (ifExpr->hasTrailing()) {
      if (isIfExpression(ifExpr->getTrailing()))
        return 1 + getIfBlocks(ifExpr->getTrailing());
    }
  }

  return 1;
}

static bool hasTrailingBlock(std::shared_ptr<ast::Expression> expr) {
  if (isIfExpression(expr)) {
    std::shared_ptr<ast::IfExpression> ifExpr =
        std::static_pointer_cast<ast::IfExpression>(expr);
    if (ifExpr->hasTrailing()) {
      if (isIfExpression(ifExpr->getTrailing())) {
        return hasTrailingBlock(ifExpr->getTrailing());
      }
      return true;
    }
  }

  return false;
}

static std::vector<std::shared_ptr<ast::IfExpression>>
getIfExprs(std::shared_ptr<ast::IfExpression> ifExpr) {
  std::vector<std::shared_ptr<ast::IfExpression>> ifs;

  ifs.push_back(ifExpr);

  if (ifExpr->hasTrailing()) {
    if (isIfExpression(ifExpr->getTrailing())) {
      std::shared_ptr<ast::IfExpression> ifxpr =
          std::static_pointer_cast<ast::IfExpression>(ifExpr->getTrailing());

      std::vector<std::shared_ptr<ast::IfExpression>> ifs = getIfExprs(ifxpr);
      for (auto ifS : ifs)
        ifs.push_back(ifS);
    }
  }

  return ifs;
}

std::optional<std::shared_ptr<ast::Expression>>
getTrailingBlock(std::shared_ptr<ast::IfExpression> ifExpr) {}

mlir::Value
ModuleBuilder::emitIfExpression(std::shared_ptr<ast::IfExpression> ifExpr) {
  std::vector<mlir::Value> conditions;
  std::vector<mlir::Block *> blocks;

  mlir::Block *currentBlock = builder.getBlock();

  uint32_t ifBlocks = getIfBlocks(ifExpr);
  bool trailingBlock = hasTrailingBlock(ifExpr);

  std::vector<std::shared_ptr<ast::IfExpression>> ifs = getIfExprs(ifExpr);

  for (auto& ifExprSharedPtr: ifs)
    conditions.push_back(emitExpression(ifExprSharedPtr->getCondition()));


  mlir::Block *ifBlock = builder.createBlock(currentBlock);
  mlir::Value blockValue = emitExpression(ifExpr->getBlock());

  mlir::Block *elseBlock = builder.createBlock(currentBlock);
  mlir::Value elseValue = emitExpression(ifExpr->getTrailing());

  // builder.create<Mir::CondBranchOp>(getLocation(ifExpr->getLocation()));

  assert(false);
}

} // namespace rust_compiler
