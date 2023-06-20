#include "AST/Statement.h"

#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/ItemDeclaration.h"
#include "Coercion.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkStatement(std::shared_ptr<ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::EmptyStatement: {
    break;
  }
  case ast::StatementKind::ItemDeclaration: {
    return checkItemDeclaration(
        std::static_pointer_cast<ItemDeclaration>(stmt).get());
  }
  case ast::StatementKind::LetStatement: {
    return checkLetStatement(std::static_pointer_cast<LetStatement>(stmt));
  }
  case ast::StatementKind::ExpressionStatement: {
    return checkExpressionStatement(
        std::static_pointer_cast<ExpressionStatement>(stmt));
  }
  case ast::StatementKind::MacroInvocationSemi: {
    assert(false && "to be implemented");
    break;
  }
  }
  assert(false);
}

TyTy::BaseType *TypeResolver::checkExpressionStatement(
    std::shared_ptr<ast::ExpressionStatement> exprStmt) {
  switch (exprStmt->getKind()) {
  case ExpressionStatementKind::ExpressionWithoutBlock: {
    return checkExpressionWithoutBlock(
        std::static_pointer_cast<ExpressionWithoutBlock>(
            exprStmt->getWithoutBlock()));
  }
  case ExpressionStatementKind::ExpressionWithBlock: {
    return checkExpressionWithBlock(
        std::static_pointer_cast<ExpressionWithBlock>(
            exprStmt->getWithBlock()));
  }
  }
}

TyTy::BaseType *
TypeResolver::checkLetStatement(std::shared_ptr<ast::LetStatement> let) {
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern = let->getPattern();

  TyTy::BaseType *initExprType = nullptr;
  Location initExprTypeLocation = {let->getLocation()};

  if (let->hasInit()) {
    initExprType = checkExpression(let->getInit());
    initExprTypeLocation = let->getInit()->getLocation();
    if (initExprType->getKind() == TyTy::TypeKind::Error)
      return nullptr;

    initExprType->appendReference(let->getNodeId());
  }

  TyTy::BaseType *specifiedType = nullptr;
  Location specifiedTypeLocation = {let->getLocation()};

  if (let->hasType()) {
    specifiedType = checkType(let->getType());
    specifiedTypeLocation = let->getType()->getLocation();
  }

  [[maybe_unused]] TyTy::BaseType *elseExprType = nullptr;
  if (let->hasElse()) {
    elseExprType = checkExpression(let->getElse());
  }

  // FIXME: elseExprType

  // let x: i32 = 5;
  if (specifiedType != nullptr && initExprType != nullptr) {
    coercion(let->getNodeId(),
             TyTy::WithLocation(specifiedType, specifiedTypeLocation),
             TyTy::WithLocation(initExprType, initExprTypeLocation),
             let->getLocation(), tcx);

    checkPattern(pattern, specifiedType);
  } else {
    if (specifiedType != nullptr) {
      // let x: i32;
      checkPattern(pattern, specifiedType);
    } else if (initExprType != nullptr) {
      // let x = 5;
      checkPattern(pattern, initExprType);
    } else {
      // let x;
      TyTy::BaseType *inferType =
          new TyTy::InferType(let->getNodeId(), TyTy::InferKind::General,
                              TypeHint::unknown(), let->getLocation());
      checkPattern(pattern, inferType);
    }
  }

  return TyTy::TupleType::getUnitType(let->getNodeId());
}

  TyTy::BaseType* TypeResolver::checkItemDeclaration(ast::ItemDeclaration *item) {
  if (item->hasVisItem()) {
    checkVisItem(item->getVisItem());
  } else if (item->hasMacroItem()) {
    assert(false);
  }

  return TyTy::TupleType::getUnitType(item->getNodeId());
}

} // namespace rust_compiler::sema::type_checking
