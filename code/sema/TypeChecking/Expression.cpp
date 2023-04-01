#include "AST/Expression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/BlockExpression.h"
#include "AST/ClosureExpression.h"
#include "AST/ExpressionStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/Statement.h"
#include "TyTy.h"
#include "TypeChecking.h"
#include "Unification.h"

#include "../ReturnExpressionSearcher.h"

#include <cassert>
#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkExpression(std::shared_ptr<ast::Expression> expr) {
  TyTy::BaseType *infered = nullptr;
  switch (expr->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    infered = checkExpressionWithBlock(
        std::static_pointer_cast<ast::ExpressionWithBlock>(expr));
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    infered = checkExpressionWithoutBlock(
        std::static_pointer_cast<ast::ExpressionWithoutBlock>(expr));
    break;
  }
  }
  infered->setReference(expr->getNodeId());
  tcx->insertType(expr->getIdentity(), infered);

  return infered;
}

TyTy::BaseType *TypeResolver::checkExpressionWithBlock(
    std::shared_ptr<ast::ExpressionWithBlock> withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ast::ExpressionWithBlockKind::BlockExpression: {
    return checkBlockExpression(
        std::static_pointer_cast<ast::BlockExpression>(withBlock));
  }
  case ast::ExpressionWithBlockKind::UnsafeBlockExpression: {
    assert(false && "to be implemented");
  }
  case ast::ExpressionWithBlockKind::LoopExpression: {
    assert(false && "to be implemented");
  }
  case ast::ExpressionWithBlockKind::IfExpression: {
    assert(false && "to be implemented");
  }
  case ast::ExpressionWithBlockKind::IfLetExpression: {
    assert(false && "to be implemented");
  }
  case ast::ExpressionWithBlockKind::MatchExpression: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    return checkLiteral(std::static_pointer_cast<LiteralExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    return checkPathExpression(
        std::static_pointer_cast<PathExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    return checkOperatorExpression(
        std::static_pointer_cast<OperatorExpression>(woBlock));
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
    return checkClosureExpression(
        std::static_pointer_cast<ClosureExpression>(woBlock));
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
    return checkReturnExpression(
        std::static_pointer_cast<ReturnExpression>(woBlock));
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be implemented");
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkBlockExpression(
    std::shared_ptr<ast::BlockExpression> block) {
  // assert(false && "to be implemented");
  Statements stmts = block->getExpressions();

  for (auto &s : stmts.getStmts()) {
    if (s->getKind() == StatementKind::ItemDeclaration)
      continue;

    checkStatement(s);
  }

  for (auto &s : stmts.getStmts()) {
    if (s->getKind() == StatementKind::ItemDeclaration)
      continue;

    TyTy::BaseType *stmtType = checkStatement(s);
    if (!stmtType) {
      llvm::errs() << "failed to resolve type: " << s->getLocation().toString()
                   << "\n";
      // report error
      return new TyTy::ErrorType(block->getNodeId());
    }

    if (s->getKind() == StatementKind::ExpressionStatement) {
      std::shared_ptr<ExpressionStatement> es =
          std::static_pointer_cast<ExpressionStatement>(s);
      if (es->getKind() == ExpressionStatementKind::ExpressionWithBlock) {
        // FIXME unify
      }
    }
  }

  if (stmts.hasTrailing())
    return checkExpression(stmts.getTrailing());
  else if (containsReturnExpression(block.get()))
    return new TyTy::NeverType(block->getNodeId());

  // FIXME
  return TyTy::TupleType::getUnitType(block->getNodeId());
}

TyTy::BaseType *TypeResolver::checkOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> op) {
  switch (op->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::DereferenceExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::NegationExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    return checkArithmeticOrLogicalExpression(
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(op));
  }
  case OperatorExpressionKind::ComparisonExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::TypeCastExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::AssignmentExpression: {
    assert(false && "to be implemented");
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith) {
  TyTy::BaseType *lhs = checkExpression(arith->getLHS());
  TyTy::BaseType *rhs = checkExpression(arith->getRHS());

  assert(lhs->getKind() != TyTy::TypeKind::Error);
  assert(rhs->getKind() != TyTy::TypeKind::Error);

  // FIXME resolveIOperatorOverload
  bool operatorOverloaded =
      resolveOperatorOverload(arith->getKind(), arith, lhs, rhs);
  assert(operatorOverloaded);

  if (!(validateArithmeticType(arith->getKind(), lhs) and
        validateArithmeticType(arith->getKind(), rhs))) {
    // report error
    llvm::errs() << arith->getLocation().toString()
                 << "cannot apply this operator to the given types"
                 << "\n";
    return new TyTy::ErrorType(arith->getNodeId());
  }

  switch (arith->getKind()) {
  case ArithmeticOrLogicalExpressionKind::LeftShift:
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    assert(false && "to be implemented");
  }
  default: {
    return unifyWithSite(
        arith->getNodeId(),
        TyTy::WithLocation(lhs, arith->getLHS()->getLocation()),
        TyTy::WithLocation(lhs, arith->getLHS()->getLocation()),
        arith->getLocation());
  }
  }
}

TyTy::BaseType *TypeResolver::checkReturnExpression(
    std::shared_ptr<ast::ReturnExpression> ret) {
  Location loc = ret->hasTailExpression() ? ret->getExpression()->getLocation()
                                          : ret->getLocation();

  TyTy::BaseType *functionReturnTye = peekReturnType();

  assert(functionReturnTye != nullptr);

  TyTy::BaseType *ty = nullptr;
  if (ret->hasTailExpression()) {
    ty = checkExpression(ret->getExpression());
  } else {
    ty = TyTy::TupleType::getUnitType(ret->getNodeId());
  }

  TyTy::BaseType *infered =
      unify(ret->getNodeId(), TyTy::WithLocation(functionReturnTye),
            TyTy::WithLocation(ty, loc), ret->getLocation());

  return new TyTy::NeverType(ret->getNodeId());
}

bool TypeResolver::validateArithmeticType(
    ArithmeticOrLogicalExpressionKind kind, TyTy::BaseType *t) {
  // https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators
  // reconsider with trait support
  switch (kind) {
  case ArithmeticOrLogicalExpressionKind::Addition:
  case ArithmeticOrLogicalExpressionKind::Subtraction:
  case ArithmeticOrLogicalExpressionKind::Multiplication:
  case ArithmeticOrLogicalExpressionKind::Division:
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    // integer or float
    switch (t->getKind()) {
    case TyTy::TypeKind::Int:
    case TyTy::TypeKind::Uint:
    case TyTy::TypeKind::Float:
    case TyTy::TypeKind::USize:
    case TyTy::TypeKind::ISize:
      return true;
    case TyTy::TypeKind::Inferred:
      return (static_cast<TyTy::InferType *>(t)->getInferredKind() ==
              TyTy::InferKind::Integral) or
             (static_cast<TyTy::InferType *>(t)->getInferredKind() ==
              TyTy::InferKind::Float);
    default:
      return false;
    }
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd:
  case ArithmeticOrLogicalExpressionKind::BitwiseOr:
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    // integer or bools
    switch (t->getKind()) {
    case TyTy::TypeKind::Int:
    case TyTy::TypeKind::Uint:
    case TyTy::TypeKind::USize:
    case TyTy::TypeKind::ISize:
    case TyTy::TypeKind::Bool:
      return true;
    case TyTy::TypeKind::Inferred:
      return static_cast<TyTy::InferType *>(t)->getInferredKind() ==
             TyTy::InferKind::Integral;
    default:
      return false;
    }
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift:
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    // integers
    switch (t->getKind()) {
    case TyTy::TypeKind::Int:
    case TyTy::TypeKind::Uint:
    case TyTy::TypeKind::USize:
    case TyTy::TypeKind::ISize:
      return true;
    case TyTy::TypeKind::Inferred:
      return static_cast<TyTy::InferType *>(t)->getInferredKind() ==
             TyTy::InferKind::Integral;
    default:
      return false;
    }
  }
  }
}

bool TypeResolver::resolveOperatorOverload(
    ArithmeticOrLogicalExpressionKind,
    std::shared_ptr<ast::ArithmeticOrLogicalExpression>, TyTy::BaseType *,
    TyTy::BaseType *) {
  // assert(false);
  //  FIXME
  return true;
}

} // namespace rust_compiler::sema::type_checking
