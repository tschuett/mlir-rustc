#include "TyTy.h"
#include "TypeChecking.h"

#include <optional>

namespace rust_compiler::sema::type_checking {

void TypeResolver::checkFunction(std::shared_ptr<ast::Function> f) {
  assert(false && "to be implemented");

  // generics

  if (f->hasWhereClause())
    checkWhereClause(f->getWhereClause());

  TyTy::BaseType *retType = nullptr;
  if (f->hasReturnType()) {
    std::optional<TyTy::BaseType *> resolved = checkType(f->getReturnType());
    if (!resolved) {
      // report error
    }
    
    
  } else {
    retType == TyTy::TupleType::getUnitType(f->getNode());
  }




  checkExpression(f->getBody());
}

} // namespace rust_compiler::sema::type_checking
