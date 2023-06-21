#include "AST/Expression.h"

#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "AST/QualifiedPathInExpression.h"
#include "Basic/Ids.h"
#include "CrateBuilder/CrateBuilder.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

namespace rust_compiler::crate_builder {

std::optional<mlir::Value> CrateBuilder::emitExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    return emitExpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    return emitExpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
    break;
  }
  }
}

mlir::Value
CrateBuilder::emitMethodCallExpression(ast::MethodCallExpression *expr) {
  assert(false);
}

void CrateBuilder::emitReturnExpression(ast::ReturnExpression *expr) {
  if (expr->hasTailExpression()) {
    std::optional<mlir::Value> result =
        emitExpression(expr->getExpression().get());
    if (result) {
      builder.create<mlir::func::ReturnOp>(getLocation(expr->getLocation()),
                                           *result);
      return;
    }
    llvm::errs() << "emitExpression in emitReturnExpression failed"
                 << "\n";
    return;
  }
  builder.create<mlir::func::ReturnOp>(getLocation(expr->getLocation()));
}

mlir::Value CrateBuilder::emitPathInExpression(ast::PathInExpression *expr) {
  std::optional<TyTy::BaseType *> type = tyCtx->lookupType(expr->getNodeId());
  if (type) {
  }
  std::optional<basic::NodeId> id = tyCtx->lookupName(expr->getNodeId());
  if (id) {
    auto it = symbolTable.begin(*id);
    if (it != symbolTable.end()) {
      return *it;
    }
    auto it2 = allocaTable.begin(*id);
    if (it2 != allocaTable.end()) {
      return builder.create<mlir::memref::LoadOp>(
          getLocation(expr->getLocation()), *it2);
    }
  } else {
    std::vector<PathExprSegment> segs = expr->getSegments();
    for (size_t i = 0; i < segs.size(); ++i)
      llvm::errs() << segs[i].getIdent().toString() << "\n";
  }
  assert(false);
}

mlir::Value CrateBuilder::emitQualifiedPathInExpression(
    ast::QualifiedPathInExpression *expr) {
  assert(false);
}

mlir::Value CrateBuilder::emitPathExpression(ast::PathExpression *expr) {
  switch (expr->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression: {
    return emitPathInExpression(static_cast<ast::PathInExpression *>(expr));
  }
  case PathExpressionKind::QualifiedPathInExpression: {
    return emitQualifiedPathInExpression(
        static_cast<ast::QualifiedPathInExpression *>(expr));
  }
  }
}

} // namespace rust_compiler::crate_builder
