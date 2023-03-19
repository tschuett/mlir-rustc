#include "ADT/CanonicalPath.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "Substitutions.h"
#include "TyTy.h"
#include "TypeChecking.h"

#include <optional>
#include <utility>
#include <vector>

using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

void TypeResolver::checkFunction(std::shared_ptr<ast::Function> f) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;

  // generics
  if (f->hasGenericParams())
    checkGenericParams(f->getGenericParams(), substitutions);

  if (f->hasWhereClause())
    checkWhereClause(f->getWhereClause());

  TyTy::BaseType *retType = nullptr;
  if (f->hasReturnType()) {
    TyTy::BaseType *retType = checkType(f->getReturnType());
    if (retType->getKind() == TyTy::TypeKind::Error) {
      // report error
    }

    retType->setReference(f->getReturnType()->getNodeId());
  } else {
    retType = TyTy::TupleType::getUnitType(f->getNodeId());
  }

  FunctionParameters parameters = f->getParams();

  assert(!parameters.hasSelfParam());

  std::vector<
      std::pair<std::shared_ptr<patterns::PatternNoTopAlt>, TyTy::BaseType *>>
      params;
  for (auto &param : parameters.getParams()) {
    switch (param.getKind()) {
    case FunctionParamKind::Pattern: {
      FunctionParamPattern pattern = param.getPattern();
      assert(pattern.hasType() && "to be implemented");
      TyTy::BaseType *paramType = checkType(pattern.getType());
      params.push_back({pattern.getPattern(), paramType});
      tcx->insertType(param.getIdentity(), paramType);
      checkPattern(pattern.getPattern(), paramType);
      break;
    }
    case FunctionParamKind::DotDotDot: {
      assert(false && "to be implemented");
    }
    case FunctionParamKind::Type: {
      assert(false && "to be implemented");
    }
    }

    std::optional<adt::CanonicalPath> path =
        tcx->lookupCanonicalPath(f->getNodeId());
    assert(path.has_value());
  }

  TyTy::BaseType *bodyType = checkExpression(f->getBody());

    assert(false && "to be implemented");

}

} // namespace rust_compiler::sema::type_checking
