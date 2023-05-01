#include "ADT/CanonicalPath.h"
#include "AST/ClosureParameters.h"
#include "Basic/Ids.h"
#include "Coercion.h"
#include "Session/Session.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include <optional>
#include <set>
#include <vector>

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *TypeResolver::checkClosureExpression(
    std::shared_ptr<ast::ClosureExpression> closure) {
  assert(false && "to be implemented");

  TypeCheckContextItem &currentContext = peekContext();
  TyTy::FunctionType *currentFunction = currentContext.getContextType();

  std::optional<GenericParams> genericParams =
      currentFunction->getGenericParams();

  tyctx::TypeIdentity ident = {
      adt::CanonicalPath::newSegment(closure->getNodeId(),
                                     currentFunction->getIdentifier()),
      closure->getLocation()};

  std::vector<TyTy::TypeVariable> parameterTypes;
  if (closure->hasParameters()) {

    ClosureParameters parameters = closure->getParameters();
    for (ClosureParam &param : parameters.getParameters()) {
      if (param.hasType()) {
        TyTy::BaseType *paramType = checkType(param.getType());

        parameterTypes.push_back(TyTy::TypeVariable(paramType->getReference()));

        checkPattern(param.getPattern(), paramType);
      } else {

        TyTy::TypeVariable paramType =
            TyTy::TypeVariable::getImplicitInferVariable(param.getLocation());
        parameterTypes.push_back(paramType);

        checkPattern(param.getPattern(), paramType.getType());
      }
    }
  }

  NodeId implicitArgsId = basic::getNextNodeId();
  TyTy::TupleType *closureArgs = new TyTy::TupleType(
      implicitArgsId, closure->getLocation(), parameterTypes);

  tcx->insertImplicitType(implicitArgsId, closureArgs);

  Location resultTypeLocation = closure->hasReturnType()
                                    ? closure->getReturnType()->getLocation()
                                    : closure->getLocation();

  TyTy::TypeVariable resultType =
      closure->hasReturnType()
          ? TyTy::TypeVariable(
                checkType(closure->getReturnType())->getReference())
          : TyTy::TypeVariable::getImplicitInferVariable(
                closure->getLocation());

  TyTy::BaseType *closureType = checkExpression(closure->getBody());

  coercionWithSite(
      closure->getNodeId(),
      TyTy::WithLocation(resultType.getType(), resultTypeLocation),
      TyTy::WithLocation(closureType, closure->getBody()->getLocation()),
      closure->getLocation());

  std::set<NodeId> captures = tcx->getCaptures(closure->getNodeId());

  return new TyTy::ClosureType(closure->getNodeId(), closure->getLocation(),
                               ident, closureArgs, resultType,
                               genericParams, captures);

  // FIXME
}

} // namespace rust_compiler::sema::type_checking
