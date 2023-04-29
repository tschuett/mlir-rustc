#pragma once

#include "AST/GenericArgs.h"
#include "AST/TypeParam.h"
#include "Lexer/Identifier.h"

#include <map>
#include <string>

namespace rust_compiler::sema::type_checking {
class TypeResolver;
}

namespace rust_compiler::tyctx::TyTy {

class BaseType;

class ParamType;

/// A TypeParam and a type parameter.
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

/// A type.
class SubstitutionArg {
public:
  SubstitutionArg(SubstitutionParamMapping *, BaseType *) {}
};

/// A list of
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
  std::map<lexer::Identifier, BaseType *> bindingArgs;
  std::vector<SubstitutionArg> mappings;
  bool errorFlag = false;
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html
/// aka SubstsRef
class SubstitutionsReference {
public:
  SubstitutionsReference(
      const std::vector<TyTy::SubstitutionParamMapping> &substitutions)
      : substitutions(substitutions) {}

  std::vector<SubstitutionParamMapping> getSubstitutions() const {
    return substitutions;
  }

  SubstitutionArgumentMappings getSubstitutionArguments() const {
    return usedArguments;
  }

  BaseType *inferSubstitutions(Location);
  SubstitutionArgumentMappings
  getMappingsFromGenericArgs(const ast::GenericArgs &,
                             sema::type_checking::TypeResolver *);
  BaseType *handleSubstitutions(SubstitutionArgumentMappings);

  bool needsSubstitution() const;

protected:
  std::string substToString() const;

private:
  std::vector<SubstitutionParamMapping> substitutions;
  SubstitutionArgumentMappings usedArguments;

  size_t getNumberOfTypeParams(const ast::GenericArgs &args);
};

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *);

} // namespace rust_compiler::tyctx::TyTy
