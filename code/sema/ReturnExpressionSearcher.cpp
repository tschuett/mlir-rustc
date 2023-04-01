#include "ReturnExpressionSearcher.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayElements.h"
#include "AST/ArrayExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/AsyncBlockExpression.h"
#include "AST/AwaitExpression.h"
#include "AST/BlockExpression.h"
#include "AST/BorrowExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/ErrorPropagationExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/GroupedExpression.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LabelBlockExpression.h"
#include "AST/LazyBooleanExpression.h"
#include "AST/LoopExpression.h"
#include "AST/MatchExpression.h"
#include "AST/NegationExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PredicateLoopExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "AST/Scrutinee.h"
#include "AST/Statement.h"
#include "AST/TypeCastExpression.h"
#include "AST/UnsafeBlockExpression.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void ReturnExpressionSearcher::visitMatchExpression(
    ast::MatchExpression *match) {
  Scrutinee scrut = match->getScrutinee();
  visitExpression(scrut.getExpression().get());

  std::vector<std::pair<MatchArm, std::shared_ptr<Expression>>> arms =
      match->getMatchArms().getArms();

  for (auto& arm: arms) {
    visitExpression(arm.second.get());
    if (arm.first.hasGuard()) {
      MatchArmGuard guard = arm.first.getGuard();
      visitExpression(guard.getGuard().get());
    }
  }
}

void ReturnExpressionSearcher::visitIfLetExpression(IfLetExpression *stmt) {
  visitExpression(stmt->getBlock().get());
  switch (stmt->getKind()) {
  case IfLetExpressionKind::NoElse: {
    break;
  }
  case IfLetExpressionKind::ElseBlock: {
    visitExpression(stmt->getTailBlock().get());
    break;
  }
  case IfLetExpressionKind::ElseIf: {
    visitExpression(stmt->getIf().get());
    break;
  }
  case IfLetExpressionKind::ElseIfLet: {
    visitExpression(stmt->getIfLet().get());
    break;
  }
  }
}

void ReturnExpressionSearcher::visitArrayExpression(ast::ArrayExpression *arr) {
  if (!arr->hasArrayElements())
    return;

  ArrayElements elements = arr->getArrayElements();
  switch (elements.getKind()) {
  case ArrayElementsKind::List: {
    std::vector<std::shared_ptr<Expression>> &el = elements.getElements();
    for (auto value : el)
      visitExpression(value.get());
    break;
  }
  case ArrayElementsKind::Repeated: {
    visitExpression(elements.getValue().get());
    break;
  }
  }
}

void ReturnExpressionSearcher::visitClosureExpression(
    ast::ClosureExpression *closure) {
  if (closure->hasBody())
    visitExpression(closure->getBody().get());
}

void ReturnExpressionSearcher::visitIteratorLoopExpression(
    ast::IteratorLoopExpression *loop) {
  visitExpression(loop->getBody().get());
  visitExpression(loop->getRHS().get());
}

void ReturnExpressionSearcher::visitPredicatePatternLoopExpression(
    ast::PredicatePatternLoopExpression *stmt) {
  visitExpression(stmt->getBody().get());
  Scrutinee scrut = stmt->getScrutinee();
  visitExpression(scrut.getExpression().get());
}

void ReturnExpressionSearcher::visitLoopExpression(ast::LoopExpression *stmt) {
  switch (stmt->getLoopExpressionKind()) {
  case LoopExpressionKind::InfiniteLoopExpression: {
    visitExpression(
        static_cast<InfiniteLoopExpression *>(stmt)->getBody().get());
    break;
  }
  case LoopExpressionKind::PredicateLoopExpression: {
    visitExpression(
        static_cast<PredicateLoopExpression *>(stmt)->getBody().get());
    visitExpression(
        static_cast<PredicateLoopExpression *>(stmt)->getCondition().get());
    break;
  }
  case LoopExpressionKind::PredicatePatternLoopExpression: {
    visitPredicatePatternLoopExpression(
        static_cast<PredicatePatternLoopExpression *>(stmt));
    break;
  }
  case LoopExpressionKind::IteratorLoopExpression: {
    visitIteratorLoopExpression(static_cast<IteratorLoopExpression *>(stmt));
    break;
  }
  case LoopExpressionKind::LabelBlockExpression: {
    visitExpression(static_cast<LabelBlockExpression *>(stmt)->getBody().get());
    break;
  }
  }
}

void ReturnExpressionSearcher::visitOperatorExpression(
    ast::OperatorExpression *stmt) {
  switch (stmt->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    visitExpression(
        static_cast<BorrowExpression *>(stmt)->getExpression().get());
    break;
  }
  case OperatorExpressionKind::DereferenceExpression: {
    visitExpression(static_cast<DereferenceExpression *>(stmt)->getRHS().get());
    break;
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    visitExpression(
        static_cast<ErrorPropagationExpression *>(stmt)->getLHS().get());
    break;
  }
  case OperatorExpressionKind::NegationExpression: {
    visitExpression(static_cast<NegationExpression *>(stmt)->getRHS().get());
    break;
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    visitExpression(
        static_cast<ArithmeticOrLogicalExpression *>(stmt)->getLHS().get());
    visitExpression(
        static_cast<ArithmeticOrLogicalExpression *>(stmt)->getRHS().get());
    break;
  }
  case OperatorExpressionKind::ComparisonExpression: {
    visitExpression(static_cast<ComparisonExpression *>(stmt)->getRHS().get());
    visitExpression(static_cast<ComparisonExpression *>(stmt)->getLHS().get());
    break;
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    visitExpression(static_cast<LazyBooleanExpression *>(stmt)->getRHS().get());
    visitExpression(static_cast<LazyBooleanExpression *>(stmt)->getLHS().get());
    break;
  }
  case OperatorExpressionKind::TypeCastExpression: {
    visitExpression(static_cast<TypeCastExpression *>(stmt)->getLeft().get());
    break;
  }
  case OperatorExpressionKind::AssignmentExpression: {
    visitExpression(static_cast<AssignmentExpression *>(stmt)->getRHS().get());
    visitExpression(static_cast<AssignmentExpression *>(stmt)->getLHS().get());
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    visitExpression(
        static_cast<CompoundAssignmentExpression *>(stmt)->getRHS().get());
    visitExpression(
        static_cast<CompoundAssignmentExpression *>(stmt)->getLHS().get());
    break;
  }
  }
}

void ReturnExpressionSearcher::visitExpressionWithoutBlock(
    ast::ExpressionWithoutBlock *stmt) {
  switch (stmt->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    visitOperatorExpression(static_cast<OperatorExpression *>(stmt));
    break;
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    visitExpression(
        static_cast<GroupedExpression *>(stmt)->getExpression().get());
    break;
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    visitArrayExpression(static_cast<ArrayExpression *>(stmt));
    break;
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    visitExpression(static_cast<AwaitExpression *>(stmt)->getBody().get());
    break;
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    visitClosureExpression(static_cast<ClosureExpression *>(stmt));
    break;
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    visitExpression(
        static_cast<AsyncBlockExpression *>(stmt)->getBlock().get());
    break;
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    foundReturn = true;
    break;
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    return;
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    break;
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    break;
  }
  }
}

void ReturnExpressionSearcher::visitExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock: {
    visitExpressionWithBlock(static_cast<ExpressionWithBlock *>(expr));
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    visitExpressionWithoutBlock(static_cast<ExpressionWithoutBlock *>(expr));
    break;
  }
  }
}

void ReturnExpressionSearcher::visitBlockExpression(
    ast::BlockExpression *block) {
  Statements stmts = block->getExpressions();

  for (auto stmt : stmts.getStmts()) {
    visitStatement(stmt.get());
  }

  if (stmts.hasTrailing())
    visitExpression(stmts.getTrailing().get());
}

void ReturnExpressionSearcher::visitExpressionWithBlock(
    ast::ExpressionWithBlock *stmt) {
  switch (stmt->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    visitBlockExpression(static_cast<BlockExpression *>(stmt));
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    visitExpression(
        static_cast<UnsafeBlockExpression *>(stmt)->getBlock().get());
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    visitLoopExpression(static_cast<LoopExpression *>(stmt));
    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    IfExpression *ifExpr = static_cast<IfExpression *>(stmt);
    visitExpression(ifExpr->getBlock().get());
    if (ifExpr->hasTrailing())
      visitExpression(ifExpr->getTrailing().get());
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    visitIfLetExpression(static_cast<IfLetExpression *>(stmt));
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    visitMatchExpression(static_cast<MatchExpression *>(stmt));
    break;
  }
  }
}

void ReturnExpressionSearcher::visitExpressionStatement(
    ast::ExpressionStatement *stmt) {
  switch (stmt->getKind()) {
  case ExpressionStatementKind::ExpressionWithBlock: {
    visitExpressionWithBlock(
        static_cast<ExpressionWithBlock *>(stmt->getWithBlock().get()));
    break;
  }
  case ExpressionStatementKind::ExpressionWithoutBlock: {
    visitExpressionWithoutBlock(
        static_cast<ExpressionWithoutBlock *>(stmt->getWithoutBlock().get()));
    break;
  }
  }
}

void ReturnExpressionSearcher::visitStatement(ast::Statement *stmt) {
  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
  }
  case StatementKind::ItemDeclaration: {
  }
  case StatementKind::LetStatement: {
  }
  case StatementKind::ExpressionStatement: {
    visitExpressionStatement(static_cast<ExpressionStatement *>(stmt));
  }
  case StatementKind::MacroInvocationSemi: {
  }
  }
}

bool ReturnExpressionSearcher::containsReturnExpression(
    ast::BlockExpression *block) {
  Statements stmts = block->getExpressions();

  for (auto stmt : stmts.getStmts()) {
    visitStatement(stmt.get());
  }

  if (stmts.hasTrailing())
    visitExpression(stmts.getTrailing().get());

  return foundReturn;
}

bool containsReturnExpression(ast::BlockExpression *block) {
  ReturnExpressionSearcher searcher;

  return searcher.containsReturnExpression(block);
}

} // namespace rust_compiler::sema
