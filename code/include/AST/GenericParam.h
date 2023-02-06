#pragma once

#include "AST/AST.h"
#include "AST/LifetimeParam.h"
#include "AST/OuterAttribute.h"
#include "AST/TypeParam.h"
#include "AST/ConstParam.h"

namespace rust_compiler::ast {

class GenericParam : public Node {
  std::vector<OuterAttribute> outerAttribute;
  std::variant<LifetimeParam, TypeParam, ConstParam> param;
public:
  GenericParam(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
