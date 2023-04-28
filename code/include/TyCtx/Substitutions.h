#pragma once

#include "AST/TypeParam.h"

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

private:
  bool errorFlag = false;
};

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *);

} // namespace rust_compiler::tyctx::TyTy
