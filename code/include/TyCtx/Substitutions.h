#pragma once

#include "AST/TypeParam.h"
#include "AST/GenericArgs.h"

#include <string>

namespace rust_compiler::tyctx::TyTy {

class BaseType;

class ParamType;

class SubstitutionParamMapping {
public:
  SubstitutionParamMapping(const ast::TypeParam &generic, ParamType *param)
      : param(param), generic(generic) {}

  std::string toString() const;

  bool needsSubstitution() const;

private:
  ParamType *param;
  ast::TypeParam generic;
};

class SubstitutionArgumentMappings {
public:
  std::string toString();

  static SubstitutionArgumentMappings error() {
    SubstitutionArgumentMappings mappings;
    mappings.errorFlag = true;
    return mappings;
  }

  size_t getSize() const;

private:
  bool errorFlag = false;
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html
/// aka SubstsRef
class SubstitutionsReference {
public:
  SubstitutionsReference(
      std::vector<TyTy::SubstitutionParamMapping> substitutions);

  std::vector<SubstitutionParamMapping> getSubstitutions() const {
    return substitutions;
  }

  SubstitutionArgumentMappings getSubstitutionArguments() const {
    return usedArguments;
  }

  BaseType *inferSubstitutions(Location);
  SubstitutionArgumentMappings
  getMappingsFromGenericArgs(const ast::GenericArgs &);
  BaseType *handleSubstitutions(SubstitutionArgumentMappings);

  bool needsSubstitution() const;

protected:
  std::string substToString() const;

private:
  std::vector<SubstitutionParamMapping> substitutions;
  SubstitutionArgumentMappings usedArguments;
};

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *);

} // namespace rust_compiler::tyctx::TyTy
