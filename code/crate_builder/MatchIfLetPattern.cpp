#include "AST/Expression.h"
#include "AST/OperatorExpression.h"
#include "CrateBuilder/CrateBuilder.h"

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value CrateBuilder::emitMatchIfLetPattern(ast::patterns::Pattern *pattern,
                                                ast::Expression *scrutinee) {
  assert(false && "to be implemented");
  std::vector<mlir::Value> patResults;
  std::vector<std::shared_ptr<ast::patterns::PatternNoTopAlt>> patterns =
      pattern->getPatterns();

  for (auto pat : patterns)
    patResults.push_back(emitMatchIfLetNoTopAlt(pat.get(), scrutinee));
}

mlir::Value
CrateBuilder::emitMatchIfLetNoTopAlt(ast::patterns::PatternNoTopAlt *pattern,
                                     ast::Expression *scrutinee) {
  assert(false && "to be implemented");
  if (scrutinee->getExpressionKind() ==
      ast::ExpressionKind::ExpressionWithBlock) {
    assert(false && "to be implemented");
  }

  ast::ExpressionWithoutBlock *woBlock =
      static_cast<ast::ExpressionWithoutBlock *>(scrutinee);

  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be implemented");
  }
  }
}

} // namespace rust_compiler::crate_builder
