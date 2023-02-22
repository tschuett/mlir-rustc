#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

#include <span>
#include <vector>

namespace rust_compiler::ast {

enum class SelfParamKind { ShorthandSelf, TypeSelf };

class SelfParam : public Node {
  SelfParamKind kind;
  std::vector<OuterAttribute> outerAttributes;

  std::shared_ptr<ast::SelfParam> self;

public:
  SelfParam(Location loc) : Node(loc) {}

  void setSelf(SelfParamKind _kind, std::shared_ptr<ast::SelfParam> _self) {
    kind = _kind;
    self = _self;
  }

  void setOuterAttributes(std::span<OuterAttribute> outerAttribute) {
    outerAttributes = {outerAttribute.begin(), outerAttribute.end()};
  }
};

} // namespace rust_compiler::ast
