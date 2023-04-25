#pragma once

#include "AST/AST.h"
#include "AST/ConstParam.h"
#include "AST/LifetimeParam.h"
#include "AST/OuterAttribute.h"
#include "AST/TypeParam.h"

#include <optional>
#include <span>

namespace rust_compiler::ast {

enum class GenericParamKind { LifetimeParam, TypeParam, ConstParam };

class GenericParam : public Node {
  std::vector<OuterAttribute> outerAttributes;
  GenericParamKind kind;
  std::optional<ConstParam> constParam;
  std::optional<TypeParam> typeParam;
  std::optional<LifetimeParam> lifetimeParam;

public:
  GenericParam(Location loc) : Node(loc) {}

  GenericParamKind getKind() const { return kind; };

  void setOuterAttributes(std::span<ast::OuterAttribute> outer) {
    outerAttributes = {outer.begin(), outer.end()};
  }

  void setConstParam(const ConstParam &cp) {
    kind = GenericParamKind::ConstParam;
    constParam = cp;
  }

  void setTypeParam(const TypeParam &tp) {
    kind = GenericParamKind::TypeParam;
    typeParam = tp;
  }

  void setLifetimeParam(const LifetimeParam &tp) {
    kind = GenericParamKind::LifetimeParam;
    lifetimeParam = tp;
  }

  TypeParam getTypeParam() const { return *typeParam; }
  ConstParam getConstParam() const { return *constParam; }
};

} // namespace rust_compiler::ast
