#include "AST/Statement.h"

#include "TypeChecking.h"

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
    assert(false && "to be implemented");
  }
  case ast::StatementKind::ExpressionStatement: {
    assert(false && "to be implemented");
  }
  case ast::StatementKind::MacroInvocationSemi: {
    assert(false && "to be implemented");
  }
  }
}

} // namespace rust_compiler::sema::type_checking
