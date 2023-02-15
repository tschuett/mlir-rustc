#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

#include <vector>
#include <span>

namespace rust_compiler::ast {

enum class SelfParamKind { ShorthandSelf, TypeSelf };

class SelfParam : public Node {
  SelfParamKind kind;
  std::vector<OuterAttribute> outerAttributes;

  std::shared_ptr<ast::SelfParam> self;

public:
  SelfParam(Location loc) : Node(loc) {}

  void setSelf(SelfParamKind kind, std::shared_ptr<ast::SelfParam> self);

  void setOuterAttributes(std::span<OuterAttribute> outerAttributes);
  
};

} // namespace rust_compiler::ast
