#include "ADT/CanonicalPath.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "Coercion.h"
#include "Location.h"
#include "Substitutions.h"
#include "TyCtx/NodeIdentity.h"
#include "TyTy.h"
#include "TypeChecking.h"
#include "llvm/Support/raw_ostream.h"

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
  Location returnTypeLoc = Location::getEmptyLocation();
  if (f->hasReturnType()) {
    TyTy::BaseType *retType = checkType(f->getReturnType());
    assert(retType);
    if (retType->getKind() == TyTy::TypeKind::Error) {
      // report error
      llvm::errs() << "failed to resolve return type"
                   << "\n";
    }

    retType->setReference(f->getReturnType()->getNodeId());
    returnTypeLoc = f->getReturnType()->getLocation();
  } else {
    retType = TyTy::TupleType::getUnitType(f->getNodeId());
    returnTypeLoc = f->getLocation();
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
  }

  std::optional<adt::CanonicalPath> path =
      tcx->lookupCanonicalPath(f->getNodeId());
  assert(path.has_value());

  tyctx::ItemIdentity identity = {*path, f->getLocation()};

  TyTy::FunctionType *funType = new TyTy::FunctionType(
      f->getNodeId(), f->getName(), identity, params, retType, substitutions);

  tcx->insertType(f->getIdentity(), funType);

  pushReturnType(TypeCheckContextItem(f.get()), funType->getReturnType());

  TyTy::BaseType *bodyType = checkExpression(f->getBody());
  assert(bodyType);

  coercionWithSite(f->getNodeId(), TyTy::WithLocation(retType, returnTypeLoc),
                   TyTy::WithLocation(bodyType), f->getLocation());

  popReturnType();
}

} // namespace rust_compiler::sema::type_checking
