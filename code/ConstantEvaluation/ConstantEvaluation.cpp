#include "ConstantEvaluation/ConstantEvaluation.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/ConstantItem.h"
#include "AST/Expression.h"
#include "AST/LiteralExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/PathExpression.h"
#include "AST/VisItem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <llvm/ADT/APInt.h>

using namespace rust_compiler::ast;

namespace rust_compiler::constant_evaluation {

uint64_t ConstantEvaluation::foldAsUsize(const ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    return foldAsUsize(static_cast<const ExpressionWithBlock *>(expr));
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    return foldAsUsize(static_cast<const ExpressionWithoutBlock *>(expr));
  }
  }
}

uint64_t
ConstantEvaluation::foldAsUsize(const ast::ExpressionWithBlock *withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    break;
  }
  }

  llvm::errs() << "trying to fold an expression with block"
               << "\n";
  exit(EXIT_FAILURE);
}

uint64_t
ConstantEvaluation::foldAsUsize(const ast::ExpressionWithoutBlock *expr) {
  switch (expr->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression:
    return foldAsUsize(static_cast<const LiteralExpression *>(expr));
  case ExpressionWithoutBlockKind::PathExpression:
    return foldAsUsize(static_cast<const PathExpression *>(expr));
  case ExpressionWithoutBlockKind::OperatorExpression:
    return foldAsUsize(static_cast<const OperatorExpression *>(expr));
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

  llvm::errs() << "trying to fold an expression without block"
               << "\n";
  exit(EXIT_FAILURE);
}

uint64_t ConstantEvaluation::foldAsUsize(const ast::LiteralExpression *lit) {
  if (lit->getLiteralKind() != LiteralExpressionKind::IntegerLiteral) {
    llvm::errs() << "literal is not integer"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  llvm::APInt result;

  std::string storage = lit->getValue();
  llvm::StringRef(storage).getAsInteger(10, result);

  unsigned bitWidth = result.getBitWidth();

  if (bitWidth <= 64)
    return result.getLimitedValue();

  llvm::errs() << "failed to fold literal"
               << "\n";

  exit(EXIT_FAILURE);
}

uint64_t ConstantEvaluation::foldAsUsize(const ast::PathExpression *path) {
  std::optional<basic::NodeId> resolvedPath =
      tyCtx->lookupName(path->getNodeId());

  if (resolvedPath) {
    std::optional<Owner> owner = getOwner(*resolvedPath);
    assert(owner.has_value());

    switch ((*owner).getKind()) {
    case OwnerKind::Expression: {
      assert(false);
    }
    case OwnerKind::Statement: {
      assert(false);
    }
    case OwnerKind::Item: {
      return foldAsUsize((*owner).getItem());
    }
    }
    assert(false);
    // uint64_t owned = foldAsUsizeNode(*owner);
    // return owned;
  }

  assert(false);
}

uint64_t ConstantEvaluation::foldAsUsize(const ast::Item *item) {
  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    return foldAsUsize(static_cast<const ast::VisItem *>(item));
  }
  case ItemKind::MacroItem: {
    assert(false);
  }
  }
}

uint64_t ConstantEvaluation::foldAsUsize(const ast::VisItem *vis) {
  switch (vis->getKind()) {
  case VisItemKind::Module: {
    assert(false);
  }
  case VisItemKind::ExternCrate: {
    assert(false);
  }
  case VisItemKind::UseDeclaration: {
    assert(false);
  }
  case VisItemKind::Function: {
    assert(false);
  }
  case VisItemKind::TypeAlias: {
    assert(false);
  }
  case VisItemKind::Struct: {
    assert(false);
  }
  case VisItemKind::Enumeration: {
    assert(false);
  }
  case VisItemKind::Union: {
    assert(false);
  }
  case VisItemKind::ConstantItem: {
    return foldAsUsize(static_cast<const ast::ConstantItem *>(vis));
  }
  case VisItemKind::StaticItem: {
    assert(false);
  }
  case VisItemKind::Trait: {
    assert(false);
  }
  case VisItemKind::Implementation: {
    assert(false);
  }
  case VisItemKind::ExternBlock: {
    assert(false);
  }
  }
}

uint64_t ConstantEvaluation::foldAsUsize(const ast::ConstantItem *con) {
  if (con->hasInit()) {
    uint64_t init = foldAsUsize(con->getInit().get());
    return init;
  }
  assert(false);
}

uint64_t ConstantEvaluation::foldAsUsize(const ast::OperatorExpression *ops) {
  switch (ops->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    break;
  }
  case OperatorExpressionKind::DereferenceExpression: {
    break;
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    break;
  }
  case OperatorExpressionKind::NegationExpression: {
    break;
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression:
    return foldAsUsize(static_cast<const ArithmeticOrLogicalExpression *>(ops));
  case OperatorExpressionKind::ComparisonExpression:
    return foldAsUsize(static_cast<const ComparisonExpression *>(ops));
  case OperatorExpressionKind::LazyBooleanExpression: {
    break;
  }
  case OperatorExpressionKind::TypeCastExpression: {
    break;
  }
  case OperatorExpressionKind::AssignmentExpression: {
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    break;
  }
  }

  llvm::errs() << "trying to fold an operator expression"
               << "\n";
  exit(EXIT_FAILURE);
}

uint64_t ConstantEvaluation::foldAsUsize(
    const ast::ArithmeticOrLogicalExpression *arith) {
  switch (arith->getKind()) {
  case ArithmeticOrLogicalExpressionKind::Addition: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Subtraction: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Multiplication: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Division: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    break;
  }
  }

  llvm::errs() << "trying to fold an arithmetic or logical expression"
               << "\n";
  exit(EXIT_FAILURE);
}

uint64_t
ConstantEvaluation::foldAsUsize(const ast::ComparisonExpression *comp) {
  switch (comp->getKind()) {
  case ComparisonExpressionKind::Equal: {
    break;
  }
  case ComparisonExpressionKind::NotEqual: {
    break;
  }
  case ComparisonExpressionKind::GreaterThan: {
    break;
  }
  case ComparisonExpressionKind::LessThan: {
    break;
  }
  case ComparisonExpressionKind::GreaterThanOrEqualTo: {
    break;
  }
  case ComparisonExpressionKind::LessThanOrEqualTo: {
    break;
  }
  }

  llvm::errs() << "trying to fold an comparison expression"
               << "\n";
  exit(EXIT_FAILURE);
}

} // namespace rust_compiler::constant_evaluation
