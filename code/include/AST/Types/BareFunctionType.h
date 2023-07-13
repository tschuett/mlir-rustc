#pragma once

#include "AST/Abi.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/ForLifetimes.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast::types {

using namespace rust_compiler::lexer;

class MaybeNamedParam : public Node {
  std::vector<OuterAttribute> outerAttributes;

  std::shared_ptr<ast::types::TypeExpression> type;
  Identifier identifier;
  bool underscore = false;

public:
  MaybeNamedParam(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }
  void setType(std::shared_ptr<ast::types::TypeExpression> tt) { type = tt; }
  void setIdentifier(const Identifier &i) { identifier = i; }
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
  bool trailingComma = false;

public:
  MaybeNamedFunctionParameters(Location loc)
      : FunctionParametersMaybeNamedVariadic(
            loc, FunctionParametersMaybeNamedVariadicKind::
                     MaybeNamedFunctionParameters){};

  bool hasTrailingcomma() const { return trailingComma; }

  void addParameter(const MaybeNamedParam &param) { params.push_back(param); }
  void setTrailingComma() { trailingComma = true; }
};

class MaybeNamedFunctionParametersVariadic
    : public FunctionParametersMaybeNamedVariadic {
  std::vector<MaybeNamedParam> params;
  std::vector<OuterAttribute> outerAttributes;

public:
  MaybeNamedFunctionParametersVariadic(Location loc)
      : FunctionParametersMaybeNamedVariadic(
            loc, FunctionParametersMaybeNamedVariadicKind::
                     MaybeNamedFunctionParametersVariadic) {}

  void addParameter(const MaybeNamedParam &param) { params.push_back(param); }
  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }
};

class BareFunctionReturnType : public Node {
  std::shared_ptr<ast::types::TypeExpression> noBounds;

public:
  BareFunctionReturnType(Location loc) : Node(loc) {}

  void setType(std::shared_ptr<ast::types::TypeExpression> no) {
    noBounds = no;
  }

  std::shared_ptr<ast::types::TypeExpression> getType() const {
    return noBounds;
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
  bool hasAbi() const { return abi.has_value(); }
  Abi getAbi() const { return *abi;}
};

class BareFunctionType : public TypeNoBounds {
  std::optional<ForLifetimes> forLifetimes;
  FunctionTypeQualifiers qualifiers;
  std::optional<std::shared_ptr<FunctionParametersMaybeNamedVariadic>> params;

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
  void
  setParameters(std::shared_ptr<FunctionParametersMaybeNamedVariadic> par) {
    params = par;
  }

  FunctionTypeQualifiers getQualifiers() const { return qualifiers; }
  bool hasParameters() const { return params.has_value(); }
  std::shared_ptr<FunctionParametersMaybeNamedVariadic> getParameters() const {
    return *params;
  }

  bool hasReturnType() const { return returnType.has_value(); }
  BareFunctionReturnType getReturnType() const { return *returnType; }
};

} // namespace rust_compiler::ast::types
