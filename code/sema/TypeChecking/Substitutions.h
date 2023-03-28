#pragma once

// #include "TyTy.h"

namespace rust_compiler::sema::type_checking::TyTy {

class BaseType;

class SubstitutionParamMapping {};

class SubstitutionArgumentMappings {};

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *);

} // namespace rust_compiler::sema::type_checking::TyTy
