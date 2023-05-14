#include "TyCtx/Substitutions.h"

#include "AST/GenericArg.h"
#include "AST/GenericArgsBinding.h"
#include "Lexer/Identifier.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"

#include <map>

using namespace rust_compiler::ast;

namespace rust_compiler::tyctx::TyTy {

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *) {
  assert(false);
}

std::string SubstitutionParamMapping::toString() const {
  if (param == nullptr)
    return "nullptr";

  return param->toString();
}

bool SubstitutionParamMapping::needSubstitution() const {
  return !param->isConcrete();
}

BaseType *SubstitutionRef::inferSubstitions(Location) { assert(false); }

std::string SubstitutionRef::substToString() const {
  std::string buffer;
  for (size_t i = 0; i < substitutions.size(); i++) {
    const SubstitutionParamMapping &sub = substitutions[i];
    buffer += sub.toString();

    if ((i + 1) < substitutions.size())
      buffer += ", ";
  }

  return buffer.empty() ? "" : "<" + buffer + ">";
}

bool SubstitutionRef::needsSubstitution() const {
  for (auto &sub : substitutions)
    if (sub.needSubstitution())
      return true;
  return false;
}

std::vector<SubstitutionParamMapping> SubstitutionRef::cloneSubsts() const {
  std::vector<SubstitutionParamMapping> clone;

  for (auto &sub : substitutions)
    clone.push_back(sub.clone());

  return clone;
}

SubstitutionParamMapping SubstitutionParamMapping::clone() const {
  return SubstitutionParamMapping(generic,
                                  static_cast<ParamType *>(param->clone()));
}

bool SubstitutionParamMapping::fillParamType(
    SubstitutionArgumentMappings &mappings, Location loc) {
  std::optional<SubstitutionArg> arg =
      mappings.getArgumentForSymbol(getParamType());
  if (!arg)
    return true;

  BaseType *type = (*arg).getType();
  if (type->getKind() == TypeKind::Inferred)
    type->inheritBounds(*param);

  if (type->getKind() == TypeKind::Parameter) {
    param = static_cast<ParamType *>(type->clone());
  } else {
    if (!param->isImplicitSelfTrait())
      if (!param->isBoundsCompatible(*type, loc, true))
        return false;

    // -> HRTB
    for (auto &bound : param->getSpecifiedBounds())
      bound.handleSubstitions(mappings);

    param->setTypeReference(type->getReference());
    mappings.onParamSubst(*param, *arg);
  }

  return true;
}

std::optional<SubstitutionArg>
SubstitutionArgumentMappings::getArgumentForSymbol(const ParamType *symbol) {
  for (auto &mapping : mappings) {
    const SubstitutionParamMapping *parm = mapping.getParamMapping();
    const ParamType *p = parm->getParamType();

    if (p->getSymbol() == symbol->getSymbol())
      return mapping;
  }

  return std::nullopt;
}

void SubstitutionArgumentMappings::onParamSubst(
    const ParamType &p, const SubstitutionArg &a) const {
  if (paramSubstCallback == nullptr)
    return;

  paramSubstCallback(p, a);
}

SubstitutionArgumentMappings
SubstitutionRef::getMappingsFromGenericArgs(ast::GenericArgs &args) {
  std::map<lexer::Identifier, BaseType *> bindingArgument;
  std::vector<GenericArg> argsV = args.getArgs();
  size_t bindings = 0;
  for (GenericArg &arg : argsV)
    if (arg.getKind() == GenericArgKind::Binding)
      ++bindings;
  if (bindings > 0) {
    if (supportsAssociatedBindings()) {
      if (bindings > getNumberOfAssociatedBindings()) {
        // report error
        assert(false);
      }

      for (GenericArg &arg : argsV)
        if (arg.getKind() == GenericArgKind::Binding) {
          //GenericArgsBinding bind = arg.getBinding();
          //BaseType *resolved = checkType();
          assert(false);
        }
    }
  }
  assert(false);
}

} // namespace rust_compiler::tyctx::TyTy
