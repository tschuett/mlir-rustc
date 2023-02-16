#pragma once

#include "AST/GenericParams.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/ForLifetimes.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/Types.h"

#include <memory>
#include <optional>
#include <vector>

namespace rust_compiler::ast::types {

class MaybeNamedParam : public Node {
  std::vector<OuterAttribute> outerAttributes;

  // FIXME

  std::shared_ptr<ast::types::Type> type;

public:
  MaybeNamedParam(Location loc) : Node(loc) {}
};

enum class FunctionParametersMaybeNamedVariadicKind {
  MaybeNamedFunctionParameters,
  MaybeNamedFunctionParametersVariadic
};

class FunctionParametersMaybeNamedVariadic : public Node {
  FunctionParametersMaybeNamedVariadicKind kind;

public:
  FunctionParametersMaybeNamedVariadic(
      Location loc, FunctionParametersMaybeNamedVariadicKind kind)
      : Node(loc), kind(kind) {}
};

class MaybeNamedFunctionParameters
    : public FunctionParametersMaybeNamedVariadic {
  std::vector<MaybeNamedParam> params;
  bool trailingComma;

public:
  MaybeNamedFunctionParameters(Location loc)
      : FunctionParametersMaybeNamedVariadic(
            loc, FunctionParametersMaybeNamedVariadicKind::
                     MaybeNamedFunctionParameters){};
};

class MaybeNamedFunctionParametersVariadic
    : public FunctionParametersMaybeNamedVariadic {
  std::vector<MaybeNamedParam> params;
  bool trailingComma;
  std::vector<OuterAttribute> outerAttributes;

public:
  MaybeNamedFunctionParametersVariadic(Location loc)
      : FunctionParametersMaybeNamedVariadic(
            loc, FunctionParametersMaybeNamedVariadicKind::
                     MaybeNamedFunctionParametersVariadic) {}
};

class BareFunctionReturnType : public Node {
  std::shared_ptr<ast::types::TypeNoBounds> noBounds;

public:
  BareFunctionReturnType(Location loc) : Node(loc) {}
};

class FunctionTypeQualifiers : public Node {
  bool unsafe;

public:
  FunctionTypeQualifiers(Location loc) : Node(loc) {}
};

class BareFunctionType : public TypeNoBounds {
  std::optional<ForLifetimes> forLifetimes;
  FunctionTypeQualifiers qualifiers;
  std::string identifier;
  std::optional<FunctionParametersMaybeNamedVariadic> params;

  std::optional<BareFunctionReturnType> returnType;

public:
  BareFunctionType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::BareFunctionType), qualifiers(loc) {
  }
};

} // namespace rust_compiler::ast::types
