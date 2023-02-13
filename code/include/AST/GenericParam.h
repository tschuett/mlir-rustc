#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

namespace rust_compiler::ast {

enum class GenericParamKind { LifetimeParam, TypeParam, ConstParam };

class GenericParam : public Node {
  std::vector<OuterAttribute> outerAttribute;
  GenericParamKind kind;

public:
  GenericParam(Location loc, GenericParamKind kind) : Node(loc), kind(kind) {}
};

} // namespace rust_compiler::ast
