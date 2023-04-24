#pragma once

#include <string>

namespace rust_compiler::tyctx::TyTy {

class BaseType;

class ParamType;

class SubstitutionParamMapping {
public:
  std::string toString() const;

private:
  ParamType *param;
};

class SubstitutionArgumentMappings {
public:
  std::string toString();
};

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *);

} // namespace rust_compiler::tyctx::TyTy
