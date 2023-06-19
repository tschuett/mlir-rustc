#pragma once

#include "AST/GenericArgs.h"
#include "AST/TypeParam.h"
#include "Bounds.h"
#include "Lexer/Identifier.h"

#include <map>
#include <string>
#include <vector>

// FIXME
namespace rust_compiler::sema::type_checking {
class TypeResolver;
}

namespace rust_compiler::tyctx::TyTy {

class BaseType;
class ParamType;
class SubstitutionArgumentMappings;

class SubstitutionParamMapping {
public:
  SubstitutionParamMapping(const ast::TypeParam &generic,
                           TyTy::ParamType *param)
      : generic(generic), param(param) {}
  SubstitutionParamMapping clone() const;

  std::string toString() const;

  bool needSubstitution() const;

  ParamType *getParamType() const { return param; }

  bool hasDefaultType() const;
  BaseType *getDefaultType() const;

  bool fillParamType(SubstitutionArgumentMappings &substMappings, Location loc);

private:
  ast::TypeParam generic;
  ParamType *param;
};

class SubstitutionArg {
public:
  SubstitutionArg(const SubstitutionParamMapping *param, BaseType *argument)
      : param(param), argument(argument) {}

  const SubstitutionParamMapping *getParamMapping() const { return param; }
  TyTy::BaseType *getType() const { return argument; }

private:
  const SubstitutionParamMapping *param;
  TyTy::BaseType *argument;
};

using ParamSubstCallback =
    std::function<void(const ParamType &, const SubstitutionArg &)>;

class SubstitutionArgumentMappings {
public:
  SubstitutionArgumentMappings(
      std::vector<SubstitutionArg> mappings,
      std::map<lexer::Identifier, BaseType *> bindingArgs, Location loc,
      ParamSubstCallback paramSubstCallback = nullptr,
      bool traitItemFlag = false, bool errorFlag = false)
      : mappings(mappings), bindingArgs(bindingArgs), loc(loc),
        paramSubstCallback(paramSubstCallback), traitItemFlag(traitItemFlag),
        errorFlag(errorFlag) {}

  static SubstitutionArgumentMappings error() {
    return SubstitutionArgumentMappings({}, {}, Location::getEmptyLocation(),
                                        nullptr, false, true);
  }
  static SubstitutionArgumentMappings empty() {
    return SubstitutionArgumentMappings({}, {}, Location::getEmptyLocation(),
                                        nullptr, false, false);
  }

  std::optional<SubstitutionArg> getArgumentForSymbol(const ParamType *symbol);
  Location getLocation() const { return loc; }

  void onParamSubst(const ParamType &p, const SubstitutionArg &a) const;

  std::vector<SubstitutionArg> &getMappings() { return mappings; }

  const std::vector<SubstitutionArg> &getMappings() const { return mappings; }

  std::map<lexer::Identifier, BaseType *> &getBindingArgs() {
    return bindingArgs;
  }

  const std::map<lexer::Identifier, BaseType *> &getBindingArgs() const {
    return bindingArgs;
  }

  TypeBoundPredicateItem
  lookupAssociatedItem(const lexer::Identifier &search) const;

  bool isError() const { return errorFlag; }

  bool getTraitItemMode() const { return traitItemFlag; }

  size_t size() const { return mappings.size(); }
  bool isEmpty() const { return size() == 0; }

  ParamSubstCallback getSubstCb() const { return paramSubstCallback; }

private:
  std::vector<SubstitutionArg> mappings;
  std::map<lexer::Identifier, TyTy::BaseType *> bindingArgs;
  Location loc;
  ParamSubstCallback paramSubstCallback;
  bool traitItemFlag;
  bool errorFlag;
};

class SubstitutionRef {
public:
  SubstitutionRef(std::vector<SubstitutionParamMapping> substitutions,
                  SubstitutionArgumentMappings arguments)
      : substitutions(substitutions), usedArguments(arguments) {}
  virtual ~SubstitutionRef() = default;

  std::string substToString() const;

  bool needsSubstitution() const;
  BaseType *inferSubstitions(Location);
  std::vector<SubstitutionParamMapping> cloneSubsts() const;

  bool hasSubstitutions() const { return substitutions.size() > 0; }

  size_t getNumberOfSubstitutions() const { return substitutions.size(); }
  std::vector<SubstitutionParamMapping> &getSubstitutions() {
    return substitutions;
  }

  const std::vector<SubstitutionParamMapping> &getSubstitutions() const {
    return substitutions;
  }

  SubstitutionArgumentMappings &getSubstitutionArguments() {
    return usedArguments;
  }
  const SubstitutionArgumentMappings &getSubstitutionArguments() const {
    return usedArguments;
  }

  // FIXME: resolver
  SubstitutionArgumentMappings
  getMappingsFromGenericArgs(ast::GenericArgs &args,
                             sema::type_checking::TypeResolver *);

  virtual BaseType *
  handleSubstitions(SubstitutionArgumentMappings &mappings) = 0;

  virtual size_t getNumberOfAssociatedBindings() const { return 0; }

  bool supportsAssociatedBindings() const {
    return getNumberOfAssociatedBindings() > 0;
  }

  size_t minRequiredSubstitutions() const;
  size_t getNumberOfRequiredSubstitutions() const;

protected:
  std::vector<SubstitutionParamMapping> substitutions;
  SubstitutionArgumentMappings usedArguments;

private:
  TyTy::BaseType *
  resolveSubstitutionsMapper(TyTy::BaseType *base,
                             TyTy::SubstitutionArgumentMappings &mappings);
};

} // namespace rust_compiler::tyctx::TyTy
