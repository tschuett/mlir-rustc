#pragma once

#include "AST/Abi.h"
#include "AST/GenericParams.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/ForLifetimes.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast::types {

class MaybeNamedParam : public Node {
  std::vector<OuterAttribute> outerAttributes;

  std::shared_ptr<ast::types::TypeExpression> type;
  std::string identifier;
  bool underscore = false;

public:
  MaybeNamedParam(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }
  void setType(std::shared_ptr<ast::types::TypeExpression> tt) { type = tt; }
  void setIdentifier(std::string_view i) { identifier = i; }
  void setUnderscore() { underscore = true; }
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
  FunctionParametersMaybeNamedVariadicKind getKind() const { return kind; }
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

  bool hasTrailingcomma() const { return trailingComma; }
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
  bool hasTrailingcomma() const { return trailingComma; }
};

class BareFunctionReturnType : public Node {
  std::shared_ptr<ast::types::TypeExpression> noBounds;

public:
  BareFunctionReturnType(Location loc) : Node(loc) {}

  void setType(std::shared_ptr<ast::types::TypeExpression> no) {
    noBounds = no;
  }
};

class FunctionTypeQualifiers : public Node {
  bool unsafe = false;
  std::optional<Abi> abi;

public:
  FunctionTypeQualifiers(Location loc) : Node(loc) {}

  bool isUnsafe() const { return unsafe; }
  void setUnsafe() { unsafe = true; }
  void setAbi(const Abi &ab) { abi = ab; }
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

  void setForLifetimes(const ast::types::ForLifetimes &foL) {
    forLifetimes = foL;
  }
  void setQualifiers(const ast::types::FunctionTypeQualifiers &qual) {
    qualifiers = qual;
  }
  void setReturnType(const ast::types::BareFunctionReturnType &ret) {
    returnType = ret;
  }
};

} // namespace rust_compiler::ast::types
