#include "AST/Statement.h"

#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "Coercion.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkStatement(std::shared_ptr<ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::EmptyStatement: {
    break;
  }
  case ast::StatementKind::ItemDeclaration: {
    assert(false && "to be implemented");
    break;
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
  assert(false && "to be implemented");

  std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern = let->getPattern();

  TyTy::BaseType *initExprType = nullptr;
  Location initExprTypeLocation = {let->getLocation()};
  ;

  if (let->hasInit()) {
    initExprType = checkExpression(let->getInit());
    initExprTypeLocation = let->getInit()->getLocation();
    if (initExprType->getKind() == TyTy::TypeKind::Error)
      return nullptr;
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
    // elseExprType->getKind() == TyTy::TypeKind::Never;
  }

  if (specifiedType != nullptr && initExprType != nullptr) {
    coercion(let->getNodeId(),
             TyTy::WithLocation(specifiedType, specifiedTypeLocation),
             TyTy::WithLocation(initExprType, initExprTypeLocation),
             let->getLocation());

    checkPattern(pattern, specifiedType);
  } else {
    if (specifiedType != nullptr) {
      checkPattern(pattern, specifiedType);
    } else if (initExprType != nullptr) {
      checkPattern(pattern, initExprType);
    } else {
      // infer
      TyTy::BaseType *inferType =
          new TyTy::InferType(let->getNodeId(), let->getLocation());
      checkPattern(pattern, inferType);
    }
  }
}

} // namespace rust_compiler::sema::type_checking
