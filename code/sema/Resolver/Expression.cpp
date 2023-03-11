#include "AST/Expression.h"

#include "ADT/CanonicalPath.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "AST/LoopExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/Statements.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveExpression(std::shared_ptr<ast::Expression> expr,
                                 const CanonicalPath &prefix,
                                 const CanonicalPath &canonicalPrefix) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    resolveExpressionWithBlock(
        std::static_pointer_cast<ast::ExpressionWithBlock>(expr), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    resolveExpressionWithoutBlock(
        std::static_pointer_cast<ast::ExpressionWithoutBlock>(expr), prefix,
        canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveExpressionWithBlock(
    std::shared_ptr<ast::ExpressionWithBlock> withBlock,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    resolveBlockExpression(std::static_pointer_cast<BlockExpression>(withBlock),
                           prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithBlockKind::LoopExpression: {
    resolveLoopExpression(std::static_pointer_cast<LoopExpression>(withBlock), prefix,
                          canonicalPrefix);
    assert(false && "to be handled later");
  }
  case ExpressionWithBlockKind::IfExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithBlockKind::MatchExpression: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolveExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> woBlock,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    resolvePathExpression(std::static_pointer_cast<PathExpression>(woBlock));
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    resolveOperatorExpression(
        std::static_pointer_cast<OperatorExpression>(woBlock), prefix,
        canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    resolveReturnExpression(std::static_pointer_cast<ReturnExpression>(woBlock),
                            prefix, canonicalPrefix);
    break;
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be handled later");
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolveReturnExpression(
    std::shared_ptr<ast::ReturnExpression> returnExpr,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (returnExpr->hasTailExpression())
    resolveExpression(returnExpr->getExpression(), prefix, canonicalPrefix);
}

void Resolver::resolveOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> op,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
  switch (op->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::DereferenceExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::NegationExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    assert(false && "to be handled later");
    std::shared_ptr<ArithmeticOrLogicalExpression> arith =
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(op);
    resolveExpression(arith->getLHS(), prefix, canonicalPrefix);
    resolveExpression(arith->getRHS(), prefix, canonicalPrefix);
  }
  case OperatorExpressionKind::ComparisonExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::TypeCastExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::AssignmentExpression: {
    assert(false && "to be handled later");
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolvePathExpression(
    std::shared_ptr<ast::PathExpression> path) {
  switch (path->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression: {
    resolvePathInExpression(std::static_pointer_cast<PathInExpression>(path));
    break;
  }
  case PathExpressionKind::QualifiedPathInExpression: {
    resolveQualifiedPathInExpression(
        std::static_pointer_cast<QualifiedPathInExpression>(path));
    break;
  }
  }
}

void Resolver::resolvePathInExpression(std::shared_ptr<ast::PathInExpression>) {
  assert(false && "to be handled later");
}

void Resolver::resolveQualifiedPathInExpression(
    std::shared_ptr<ast::QualifiedPathInExpression>) {
  assert(false && "to be handled later");
}

void Resolver::resolveBlockExpression(
    std::shared_ptr<ast::BlockExpression> block,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  NodeId scopeNodeId = block->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getNameScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  const Statements &stmts = block->getExpressions();

  for (auto &stmt : stmts.getStmts())
      resolveStatement(stmt, prefix, canonicalPrefix,
                       CanonicalPath::createEmpty());

  if (stmts.hasTrailing())
    resolveExpression(stmts.getTrailing(), prefix, canonicalPrefix);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

void Resolver::resolveLoopExpression(
    std::shared_ptr<ast::LoopExpression>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
