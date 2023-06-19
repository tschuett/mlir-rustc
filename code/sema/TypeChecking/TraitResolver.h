#pragma once

#include "AST/Types/TypeExpression.h"
#include "TyCtx/TraitReference.h"

namespace rust_compiler::sema::TypeChecking {

class TraitResolver {
public:
  static tyctx::TyTy::TraitReference *resolve(ast::types::TypeExpression *traitPath);
};

} // namespace rust_compiler::sema::TypeChecking
