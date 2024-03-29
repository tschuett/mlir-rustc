#include "AST/ArrayElements.h"
#include "AST/ArrayExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/FieldExpression.h"
#include "AST/GroupedExpression.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/IndexEpression.h"
#include "AST/LoopExpression.h"
#include "AST/MatchArms.h"
#include "AST/MatchExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PredicateLoopExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "AST/RangeExpression.h"
#include "AST/Scrutinee.h"
#include "AST/Statement.h"
#include "AST/StructBase.h"
#include "AST/StructExprField.h"
#include "AST/StructExprStruct.h"
#include "AST/StructExprTuple.h"
#include "AST/StructExpression.h"
#include "AST/TupleExpression.h"
#include "AST/TupleIndexingExpression.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

bool Sema::isConstantExpression(ast::Expression *expr) {
  bool result = false;
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock:
    result = isConstantEpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
    break;
  case ExpressionKind::ExpressionWithoutBlock:
    result = isConstantEpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
    break;
  }
  if (result)
    expr->setConstant();
  return result;
}

bool Sema::isConstantEpressionWithoutBlock(
    ast::ExpressionWithoutBlock *woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    return isConstantOperatorExpression(
        static_cast<OperatorExpression *>(woBlock));
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    GroupedExpression *group = static_cast<GroupedExpression *>(woBlock);
    return isConstantExpression(group->getExpression().get());
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    // FIXME no Drop
    ast::ArrayExpression *array = static_cast<ArrayExpression *>(woBlock);
    ArrayElements elements = array->getArrayElements();
    switch (elements.getKind()) {
    case ArrayElementsKind::List: {
      for (auto &expr : elements.getElements())
        if (!isConstantExpression(expr.get()))
          return false;
      array->setConstant();
      return true;
    }
    case ArrayElementsKind::Repeated: {
      return isConstantExpression(elements.getValue().get()) and
             isConstantExpression(elements.getCount().get());
    }
    }
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    ast::IndexExpression *index = static_cast<IndexExpression *>(woBlock);
    return isConstantExpression(index->getArray().get()) and
           isConstantExpression(index->getIndex().get());
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    TupleExpression *tuple = static_cast<TupleExpression *>(woBlock);
    if (tuple->isUnit())
      return true;
    TupleElements elements = tuple->getElements();
    for (auto &element : elements.getElements())
      if (!isConstantExpression(element.get()))
        return false;
    tuple->setConstant();
    return true;
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    ast::TupleIndexingExpression *index =
        static_cast<TupleIndexingExpression *>(woBlock);
    return isConstantExpression(index->getTuple().get());
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    return isConstantStructExpression(static_cast<StructExpression *>(woBlock));
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    FieldExpression *field = static_cast<FieldExpression *>(woBlock);
    return isConstantExpression(field->getField().get());
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    ast::RangeExpression *range = static_cast<RangeExpression *>(woBlock);
    switch (range->getKind()) {
    case RangeExpressionKind::RangeExpr: {
      return isConstantExpression(range->getLeft().get()) and
             isConstantExpression(range->getRight().get());
    }
    case RangeExpressionKind::RangeFromExpr: {
      return isConstantExpression(range->getLeft().get());
    }
    case RangeExpressionKind::RangeToExpr: {
      return isConstantExpression(range->getRight().get());
    }
    case RangeExpressionKind::RangeFullExpr: {
      return true;
    }
    case RangeExpressionKind::RangeInclusiveExpr: {
      return isConstantExpression(range->getLeft().get()) and
             isConstantExpression(range->getRight().get());
    }
    case RangeExpressionKind::RangeToInclusiveExpr: {
      return isConstantExpression(range->getRight().get());
    }
    }
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false);
  }
  }
}

bool Sema::isConstantEpressionWithBlock(ast::ExpressionWithBlock *withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    return isConstantBlockExpression(static_cast<BlockExpression *>(withBlock));
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    return isConstantBlockExpression(static_cast<BlockExpression *>(withBlock));
  }
  case ExpressionWithBlockKind::LoopExpression: {
    return isConstantLoopExpression(static_cast<LoopExpression *>(withBlock));
  }
  case ExpressionWithBlockKind::IfExpression: {
    IfExpression *ifExpr = static_cast<IfExpression *>(withBlock);
    if (!isConstantExpression(ifExpr->getCondition().get()))
      return false;
    if (!isConstantExpression(ifExpr->getBlock().get()))
      return false;

    if (ifExpr->hasTrailing())
      if (!isConstantExpression(ifExpr->getTrailing().get()))
        return false;

    withBlock->setConstant();
    return true;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    return isConstantIfLetExpression(static_cast<IfLetExpression *>(withBlock));
  }
  case ExpressionWithBlockKind::MatchExpression: {
    MatchExpression *match = static_cast<MatchExpression *>(withBlock);
    if (!isConstantExpression(match->getScrutinee().getExpression().get()))
      return false;

    for (auto &arm : match->getMatchArms().getArms()) {
      if (!isConstantExpression(arm.second.get()))
        return false;
      if (arm.first.hasGuard())
        if (!isConstantExpression(arm.first.getGuard().getGuard().get()))
          return false;
    }

    withBlock->setConstant();
    return true;
  }
  }
}

bool Sema::isConstantOperatorExpression(ast::OperatorExpression *op) {
  switch (op->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    assert(false);
  }
  case OperatorExpressionKind::DereferenceExpression: {
    assert(false);
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    assert(false);
  }
  case OperatorExpressionKind::NegationExpression: {
    assert(false);
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    assert(false);
  }
  case OperatorExpressionKind::ComparisonExpression: {
    assert(false);
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    assert(false);
  }
  case OperatorExpressionKind::TypeCastExpression: {
    assert(false);
  }
  case OperatorExpressionKind::AssignmentExpression: {
    AssignmentExpression *assign = static_cast<AssignmentExpression *>(op);
    return isConstantExpression(assign->getLHS().get()) and
           isConstantExpression(assign->getRHS().get());
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    CompoundAssignmentExpression *compound =
        static_cast<CompoundAssignmentExpression *>(op);
    return isConstantExpression(compound->getLHS().get()) and
           isConstantExpression(compound->getRHS().get());
  }
  }
}

bool Sema::isConstantBlockExpression(ast::BlockExpression *block) {
  Statements stmts = block->getExpressions();

  for (auto &stmt : stmts.getStmts())
    if (!isConstantStatement(stmt.get()))
      return false;

  if (stmts.hasTrailing())
    return isConstantExpression(stmts.getTrailing().get());

  block->setConstant();
  return true;
}

bool Sema::isConstantStatement(ast::Statement *stmt) {
  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
    stmt->setConstant();
    return true;
  }
  case StatementKind::ItemDeclaration: {
    assert(false);
  }
  case StatementKind::LetStatement: {
    assert(false);
  }
  case StatementKind::ExpressionStatement: {
    ExpressionStatement *exprStmt = static_cast<ExpressionStatement *>(stmt);
    switch (exprStmt->getKind()) {
    case ExpressionStatementKind::ExpressionWithoutBlock: {
      return isConstantEpressionWithoutBlock(
          std::static_pointer_cast<ExpressionWithoutBlock>(
              exprStmt->getWithoutBlock())
              .get());
    }
    case ExpressionStatementKind::ExpressionWithBlock: {
      return isConstantEpressionWithBlock(
          std::static_pointer_cast<ExpressionWithBlock>(
              exprStmt->getWithBlock())
              .get());
    }
    }
  }
  case StatementKind::MacroInvocationSemi: {
    assert(false);
  }
  }
}

bool Sema::isConstantLoopExpression(ast::LoopExpression *loop) {
  switch (loop->getLoopExpressionKind()) {
  case LoopExpressionKind::InfiniteLoopExpression: {
    assert(false);
  }
  case LoopExpressionKind::PredicateLoopExpression: {
    PredicateLoopExpression *pred =
        static_cast<PredicateLoopExpression *>(loop);
    if (!isConstantExpression(pred->getCondition().get()))
      return false;
    if (!isConstantExpression(pred->getBody().get()))
      return false;
    loop->setConstant();
    return true;
  }
  case LoopExpressionKind::PredicatePatternLoopExpression: {
    PredicatePatternLoopExpression *pattern =
        static_cast<PredicatePatternLoopExpression *>(loop);
    if (!isConstantExpression(pattern->getScrutinee().getExpression().get()))
      return false;
    if (!isConstantExpression(pattern->getBody().get()))
      return false;
    loop->setConstant();
    return true;
  }
  case LoopExpressionKind::IteratorLoopExpression: {
    assert(false);
  }
  case LoopExpressionKind::LabelBlockExpression: {
    assert(false);
  }
  }
}

bool Sema::isConstantStructExpression(ast::StructExpression *se) {
  switch (se->getKind()) {
  case StructExpressionKind::StructExprStruct: {
    StructExprStruct *str = static_cast<StructExprStruct *>(se);
    if (str->hasStructBase()) {
      StructBase base = str->getStructBase();
      return isConstantExpression(base.getPath().get());
    }
    if (str->hasStructExprFields()) {
      StructExprFields fields = str->getStructExprFields();
      if (fields.hasBase()) {
        StructBase base = fields.getBase();
        if (!isConstantExpression(base.getPath().get()))
          return false;
      }
      for (StructExprField &field : fields.getFields()) {
        if (field.hasExpression())
          if (!isConstantExpression(field.getExpression().get()))
            return false;
      }
    }
    se->setConstant();
    return true;
  }
  case StructExpressionKind::StructExprTuple: {
    StructExprTuple *tuple = static_cast<StructExprTuple *>(se);
    for (auto &expr : tuple->getExpressions())
      if (!isConstantExpression(expr.get()))
        return false;
    se->setConstant();
    return true;
  }
  case StructExpressionKind::StructExprUnit: {
    assert(false);
  }
  }
}

bool Sema::isConstantIfLetExpression(ast::IfLetExpression *ifLet) {
  if (!isConstantExpression(ifLet->getScrutinee().getExpression().get()))
    return false;
  if (!isConstantExpression(ifLet->getBlock().get()))
    return false;

  switch (ifLet->getKind()) {
  case IfLetExpressionKind::NoElse: {
    ifLet->setConstant();
    return true;
  }
  case IfLetExpressionKind::ElseBlock: {
    if (!isConstantExpression(ifLet->getTailBlock().get()))
      return false;
    break;
  }
  case IfLetExpressionKind::ElseIf: {
    if (!isConstantExpression(ifLet->getIf().get()))
      return false;
    break;
  }
  case IfLetExpressionKind::ElseIfLet: {
    if (!isConstantExpression(ifLet->getIfLet().get()))
      return false;
    break;
  }
  }
  ifLet->setConstant();
  return true;
}

} // namespace rust_compiler::sema
