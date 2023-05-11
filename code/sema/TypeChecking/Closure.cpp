#include "ADT/CanonicalPath.h"
#include "AST/ClosureParameters.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Basic/Ids.h"
#include "Coercion.h"
#include "Session/Session.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include <optional>
#include <set>
#include <vector>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::basic;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *TypeResolver::checkClosureExpression(
    std::shared_ptr<ast::ClosureExpression> closure) {
  assert(false && "to be implemented");

  TypeCheckContextItem &currentContext = peekContext();
  TyTy::FunctionType *currentFunction = currentContext.getContextType();

  std::vector<TyTy::SubstitutionParamMapping> substRfs =
      currentFunction->cloneSubsts();

  tyctx::TypeIdentity ident = {
      adt::CanonicalPath::newSegment(closure->getNodeId(),
                                     currentFunction->getIdentifier()),
      closure->getLocation()};

  std::vector<TyTy::TypeVariable> parameterTypes;
  if (closure->hasParameters()) {

    ClosureParameters parameters = closure->getParameters();
    for (ClosureParam &param : parameters.getParameters()) {
      TyTy::BaseType *paramType = nullptr;
      if (param.hasType()) {
        paramType = checkType(param.getType());

      } else {
        paramType = inferClosureParam(param.getPattern().get());
        //            TyTy::TypeVariable::getImplicitInferVariable(param.getLocation());
      }
      parameterTypes.push_back(TyTy::TypeVariable(paramType->getReference()));

      checkPattern(param.getPattern(), paramType);
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
      closure->getLocation(), tcx);

  std::set<NodeId> captures = tcx->getCaptures(closure->getNodeId());

  TyTy::BaseType *result = new TyTy::ClosureType(
      closure->getNodeId(), ident, closureArgs, resultType, substRfs, captures);

  // FIXME

  return result;
}

TyTy::BaseType *
TypeResolver::inferClosureParam(patterns::PatternNoTopAlt *noTop) {
  switch (noTop->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    PatternWithoutRange *wo =
        static_cast<patterns::PatternWithoutRange *>(noTop);
    switch (wo->getWithoutRangeKind()) {
    case PatternWithoutRangeKind::LiteralPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::IdentifierPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::WildcardPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::RestPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::ReferencePattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::StructPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::TupleStructPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::TuplePattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::GroupedPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::SlicePattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::PathPattern: {
      assert(false);
    }
    case PatternWithoutRangeKind::MacroInvocation: {
      assert(false);
    }
    }
  }
  case PatternNoTopAltKind::RangePattern: {
    assert(false);
  }
  }
}

} // namespace rust_compiler::sema::type_checking
