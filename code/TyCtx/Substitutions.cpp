#include "TyCtx/Substitutions.h"

#include "AST/GenericArgsBinding.h"
#include "TyCtx/TyTy.h"
#include "../sema/TypeChecking/TypeChecking.h"

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

bool SubstitutionParamMapping::needsSubstitution() const {
  return !param->isConcrete();
}

BaseType *SubstitutionsReference::inferSubstitutions(Location) {
  assert(false);
}

size_t
SubstitutionsReference::getNumberOfTypeParams(const ast::GenericArgs &args) {
  size_t types = 0;
  for (const GenericArg &arg : args.getArgs()) {
    switch (arg.getKind()) {
    case GenericArgKind::Lifetime: {
      break;
    }
    case GenericArgKind::Type: {
      ++types;
      break;
    }
    case GenericArgKind::Const: {
      break;
    }
    case GenericArgKind::Binding: {
      break;
    }
    }
  }
  return types;
}

SubstitutionArgumentMappings
SubstitutionsReference::getMappingsFromGenericArgs(const ast::GenericArgs &args,
                                                   TypeResolver *resolver) {
  assert(false);
  std::map<Identifier, BaseType *> bindingArguments;

  // FIXME detect misbehaviour

  // size_t types = getNumberOfTypeParams(args);
  //  size_t offset = usedArguments.getSize();
  if (args.getNumberOfArgs() == substitutions.size()) {
    // FIXME
    // report error
  }

  // We can now type check based on GenericArgs. They were not
  // available when we type check based on GenericParams.
  size_t offset = 0;
  std::vector<SubstitutionArg> mappings;
  for (const GenericArg &arg : args.getArgs()) {
    switch (arg.getKind()) {
    case GenericArgKind::Lifetime: {
      // We don't type check lifetimes
      break;
    }
    case GenericArgKind::Type: {
      BaseType *type = resolver->checkType(arg.getType());
      assert(type != nullptr);
      assert(type->getKind() != TypeKind::Error);
      SubstitutionArg substArg(&substitutions[offset], type);
      mappings.push_back(substArg);
      ++offset;
      break;
    }
    case GenericArgKind::Const: {
      break;
    }
    case GenericArgKind::Binding: {
      GenericArgsBinding bind = arg.getBinding();
      BaseType *type = resolver->checkType(bind.getType());
      assert(type == nullptr);
      assert(type->getKind() != TypeKind::Error);
      bindingArguments[bind.getIdentifier()] = type;
      break;
    }
    }
  }
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
