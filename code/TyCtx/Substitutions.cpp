#include "TyCtx/Substitutions.h"

#include "AST/GenericArg.h"
#include "AST/GenericArgsBinding.h"
#include "Lexer/Identifier.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"

#include <cstddef>
#include <map>
#include <vector>
#include <llvm/ADT/StringMap.h>

// FIXME
#include "../sema/TypeChecking/TypeChecking.h"

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

BaseType *SubstitutionRef::inferSubstitions(Location loc) {
  std::vector<SubstitutionArg> args;
  std::map<Identifier, TyTy::BaseType *> argumentMappings;
  for (auto &p : getSubstitutions()) {
    if (p.needSubstitution()) {
      const Identifier& symbol = p.getParamType()->getSymbol();
      auto it = argumentMappings.find(symbol);
      if (it != argumentMappings.end()) {
        args.push_back(SubstitutionArg(&p, it->second));
      } else {
        TypeVariable inferVar = TypeVariable::getImplicitInferVariable(loc);
        args.push_back(SubstitutionArg(&p, inferVar.getType()));
      }
    } else {
      args.push_back(SubstitutionArg(&p, p.getParamType()));
    }
  }

  SubstitutionArgumentMappings inferArguments = {std::move(args), {}, loc};

  return handleSubstitions(inferArguments);
}

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
SubstitutionRef::getMappingsFromGenericArgs(ast::GenericArgs &args,
                                            TypeResolver *resolver) {
  std::map<lexer::Identifier, BaseType *> bindingArgument;
  std::vector<GenericArg> argsV = args.getArgs();
  size_t bindings = 0;
  size_t types = 0;
  for (GenericArg &arg : argsV) {
    if (arg.getKind() == GenericArgKind::Binding)
      ++bindings;
    if (arg.getKind() == GenericArgKind::Type)
      ++types;
  }
  if (bindings > 0) {
    if (supportsAssociatedBindings()) {
      if (bindings > getNumberOfAssociatedBindings()) {
        // report error
        assert(false);
      }

      for (GenericArg &arg : argsV)
        if (arg.getKind() == GenericArgKind::Binding) {
          GenericArgsBinding bind = arg.getBinding();
          BaseType *resolved = resolver->checkType(bind.getType());
          if (resolved == nullptr || resolved->getKind() == TypeKind::Error)
            return SubstitutionArgumentMappings::error();
          bindingArgument[bind.getIdentifier()] = resolved;
        }
    } else {
      // report error
      assert(false);
    }
  }

  size_t offset = usedArguments.size();

  if (types + offset > substitutions.size()) {
    // report error
    assert(false);
  }

  if (types + offset < minRequiredSubstitutions()) {
    // report error
    assert(false);
  }

  std::vector<SubstitutionArg> mappings = usedArguments.getMappings();
  for (GenericArg &arg : argsV) {
    if (arg.getKind() == GenericArgKind::Type) {
      TyTy::BaseType *resolved = resolver->checkType(arg.getType());
      if (resolved == nullptr || resolved->getKind() == TypeKind::Error) {
        // report error
        assert(false);
      }
      SubstitutionArg substArgument(&substitutions.at(offset), resolved);
      ++offset;
      mappings.push_back(substArgument);
    }
  }

  size_t leftOver =
      getNumberOfRequiredSubstitutions() - minRequiredSubstitutions();
  if (leftOver > 0) {
    for (size_t offs = mappings.size(); offs < substitutions.size(); ++offs) {
      SubstitutionParamMapping &param = substitutions[offs];
      assert(param.hasDefaultType());
      BaseType *resolved = param.getDefaultType();
      if (resolved->getKind() == TypeKind::Error)
        return SubstitutionArgumentMappings::error();

      if (!resolved->isConcrete()) {
        SubstitutionArgumentMappings temp = {mappings, bindingArgument,
                                             args.getLocation()};

        resolved = resolveSubstitutionsMapper(resolved, temp);
        if (resolved->getKind() == TypeKind::Error)
          return SubstitutionArgumentMappings::error();
      }

      SubstitutionArg substArgument(&param, resolved);
      mappings.push_back(substArgument);
    }
  }

  return SubstitutionArgumentMappings(mappings, bindingArgument,
                                      args.getLocation());
}

TyTy::BaseType *SubstitutionRef::resolveSubstitutionsMapper(
    TyTy::BaseType *base, TyTy::SubstitutionArgumentMappings &mappings) {
  assert(false);
  switch (base->getKind()) {
  case TypeKind::Bool: {
    assert(false);
  }
  case TypeKind::Char: {
    assert(false);
  }
  case TypeKind::Int: {
    assert(false);
  }
  case TypeKind::Uint: {
    assert(false);
  }
  case TypeKind::USize: {
    assert(false);
  }
  case TypeKind::ISize: {
    assert(false);
  }
  case TypeKind::Float: {
    assert(false);
  }
  case TypeKind::Closure: {
    assert(false);
  }
  case TypeKind::Function: {
    assert(false);
  }
  case TypeKind::Inferred: {
    assert(false);
  }
  case TypeKind::Never: {
    assert(false);
  }
  case TypeKind::Str: {
    assert(false);
  }
  case TypeKind::Tuple: {
    assert(false);
  }
  case TypeKind::Parameter: {
    assert(false);
  }
  case TypeKind::ADT: {
    assert(false);
  }
  case TypeKind::Array: {
    assert(false);
  }
  case TypeKind::Slice: {
    assert(false);
  }
  case TypeKind::Projection: {
    assert(false);
  }
  case TypeKind::Dynamic: {
    assert(false);
  }
  case TypeKind::PlaceHolder: {
    assert(false);
  }
  case TypeKind::FunctionPointer: {
    assert(false);
  }
  case TypeKind::RawPointer: {
    assert(false);
  }
  case TypeKind::Reference: {
    assert(false);
  }
  case TypeKind::Error: {
    assert(false);
  }
  }
}

size_t SubstitutionRef::minRequiredSubstitutions() const {
  size_t n = 0;
  for (const SubstitutionParamMapping &p : substitutions)
    if (p.needSubstitution() && !p.hasDefaultType())
      ++n;
  return n;
}

size_t SubstitutionRef::getNumberOfRequiredSubstitutions() const {
  size_t n = 0;
  for (const SubstitutionParamMapping &p : substitutions)
    if (p.needSubstitution())
      ++n;
  return n;
}

bool SubstitutionParamMapping::hasDefaultType() const {
  return generic.hasType();
}

BaseType *SubstitutionParamMapping::getDefaultType() const {
  TypeVariable var = {generic.getType()->getNodeId()};
  return var.getType();
}

} // namespace rust_compiler::tyctx::TyTy
