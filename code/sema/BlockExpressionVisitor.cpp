#include "BlockExpressionVisitor.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/BorrowExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/DereferenceExpression.h"
#include "AST/ErrorPropagationExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/Item.h"
#include "AST/LazyBooleanExpression.h"
#include "AST/LetStatement.h"
//#include "AST/MacroInvocationSemi.h"
#include "AST/NegationExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/Statement.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

static void runOperatorExpression(std::shared_ptr<ast::OperatorExpression> op,
                                  BlockExpressionVisitor *visitor) {
  switch (op->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    visitor->visitBorrowExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<BorrowExpression>(op)->getExpression());
    break;
  }
  case OperatorExpressionKind::DereferenceExpression: {
    visitor->visitDereferenceExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<DereferenceExpression>(op)->getRHS());
    break;
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    visitor->visitErrorPropagationExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<ErrorPropagationExpression>(op)->getLHS());
    break;
  }
  case OperatorExpressionKind::NegationExpression: {
    visitor->visitNegationExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<NegationExpression>(op)->getRHS());
    break;
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    visitor->visitArithmeticOrLogicalExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(op)->getLHS());
    visitor->visitExpression(
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(op)->getRHS());
    break;
  }
  case OperatorExpressionKind::ComparisonExpression: {
    visitor->visitComparisonExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<ComparisonExpression>(op)->getLHS());
    visitor->visitExpression(
        std::static_pointer_cast<ComparisonExpression>(op)->getRHS());
    break;
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    visitor->visitLazyBooleanExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<LazyBooleanExpression>(op)->getLHS());
    visitor->visitExpression(
        std::static_pointer_cast<LazyBooleanExpression>(op)->getRHS());
    break;
  }
  case OperatorExpressionKind::TypeCastExpression: {
    visitor->visitTypeCastExpression(op);
    // FIXME
    break;
  }
  case OperatorExpressionKind::AssignmentExpression: {
    visitor->visitAssignmentExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<AssignmentExpression>(op)->getLHS());
    visitor->visitExpression(
        std::static_pointer_cast<AssignmentExpression>(op)->getRHS());
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    visitor->visitCompaoundAssignmentExpression(op);
    visitor->visitExpression(
        std::static_pointer_cast<CompoundAssignmentExpression>(op)->getLHS());
    visitor->visitExpression(
        std::static_pointer_cast<CompoundAssignmentExpression>(op)->getRHS());
    break;
  }
  }
}

static void runStatement(std::shared_ptr<ast::Statement> stmt,
                         BlockExpressionVisitor *visitor) {
  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
    // empty
    break;
  }
  case StatementKind::ItemDeclaration: {
    visitor->visitItemDeclaration(
        std::static_pointer_cast<ItemDeclaration>(stmt));
    break;
  }
  case StatementKind::LetStatement: {
    visitor->visitLetStatement(std::static_pointer_cast<LetStatement>(stmt));
    break;
  }
  case StatementKind::ExpressionStatement: {
    visitor->visitExpressionStatement(
        std::static_pointer_cast<ExpressionStatement>(stmt));
    break;
  }
  case StatementKind::MacroInvocationSemi: {
    visitor->visitMacroInvocationSemi(
        std::static_pointer_cast<Statement>(stmt));
    break;
  }
  }
}

static void
runExpressionWithoutBlock(std::shared_ptr<ast::ExpressionWithoutBlock> woBlock,
                          BlockExpressionVisitor *visitor) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    visitor->visitOperatorExpression(woBlock);
    runOperatorExpression(std::static_pointer_cast<OperatorExpression>(woBlock),
                          visitor);
    break;
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
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
    break;
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
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
    break;
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    break;
  }
  }
}

void run(std::shared_ptr<ast::BlockExpression> block,
         BlockExpressionVisitor *visitor) {

  ast::Statements stmts = block->getExpressions();
  visitor->visitStatements(stmts);

  for (auto &stmt : stmts.getStmts()) {
    visitor->visitStatement(stmt);
    runStatement(stmt, visitor);
  }

  if (stmts.hasTrailing()) {
    std::shared_ptr<Expression> trail = stmts.getTrailing();
    visitor->visitExpressionWithoutBlock(
        std::static_pointer_cast<ExpressionWithoutBlock>(trail));
    runExpressionWithoutBlock(
        std::static_pointer_cast<ExpressionWithoutBlock>(trail), visitor);
  }
}

} // namespace rust_compiler::sema
