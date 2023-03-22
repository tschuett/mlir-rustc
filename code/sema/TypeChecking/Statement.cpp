#include "AST/Statement.h"

#include "TyTy.h"
#include "TypeChecking.h"

#include <memory>

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkStatement(std::shared_ptr<ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::EmptyStatement: {
    break;
  }
  case ast::StatementKind::ItemDeclaration: {
    assert(false && "to be implemented");
  }
  case ast::StatementKind::LetStatement: {
    return checkLetStatement(std::static_pointer_cast<LetStatement>(stmt));
  }
  case ast::StatementKind::ExpressionStatement: {
    assert(false && "to be implemented");
  }
  case ast::StatementKind::MacroInvocationSemi: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *
TypeResolver::checkLetStatement(std::shared_ptr<ast::LetStatement> let) {
  assert(false && "to be implemented");

  std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern = let->getPattern();

  TyTy::BaseType *initExprType = nullptr;

  if (let->hasInit()) {
    initExprType = checkExpression(let->getInit());
    if (initExprType->getKind() == TyTy::TypeKind::Error)
      return nullptr;
  }

  TyTy::BaseType *specifiedType = nullptr;

  if (let->hasType()) {
    specifiedType = checkType(let->getType());
  }

  if (specifiedType != nullptr && initExprType != nullptr) {
    coercion();

    checkPattern(pattern, specifiedType);
  } else {
    if (specifiedType != nullptr) {
      checkPattern(pattern, specifiedType);
    } else if (initExprType != nullptr) {
      checkPattern(pattern, initExprType);
    } else {
      // infer
      TyTy::BaseType *inferType = new TyTy::InferType();
      checkPattern(pattern, inferType);
    }
  }
}

} // namespace rust_compiler::sema::type_checking
