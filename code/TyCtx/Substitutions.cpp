#include "TyCtx/Substitutions.h"

#include "TyCtx/TyTy.h"

#include <map>

namespace rust_compiler::tyctx::TyTy {

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *) {
  assert(false);
}

std::string SubstitutionParamMapping::toString() const {
  if (param == nullptr)
    return "nullptr";

  return param->toString();
}

bool SubstitutionParamMapping::needsSubstitution() const {
  return !param->isConcrete();
}

BaseType *SubstitutionsReference::inferSubstitutions(Location) {
  assert(false);
}

SubstitutionArgumentMappings SubstitutionsReference::getMappingsFromGenericArgs(
    const ast::GenericArgs &args) {
  assert(false);
  std::map<std::string, BaseType *> bindingArguments;

  size_t offset = usedArguments.getSize();
//  if (args.getNumberOfArgs() + offs > substitutions.size()) {
//    // report error
//  }
}

BaseType *
SubstitutionsReference::handleSubstitutions(SubstitutionArgumentMappings) {
  assert(false);
}

std::string SubstitutionsReference::substToString() const {
  std::string buffer;
  for (size_t i = 0; i < substitutions.size(); i++) {
    const SubstitutionParamMapping &sub = substitutions[i];
    buffer += sub.toString();

    if ((i + 1) < substitutions.size())
      buffer += ", ";
  }

  return buffer.empty() ? "" : "<" + buffer + ">";
}

bool SubstitutionsReference::needsSubstitution() const {
  for (auto &sub : substitutions)
    if (sub.needsSubstitution())
      return true;
  return false;
}

} // namespace rust_compiler::tyctx::TyTy
