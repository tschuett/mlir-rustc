#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "AST/OuterAttribute.h"

#include <span>
#include <vector>

namespace rust_compiler::ast {

class ExternalItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<ast::VisItem> stat;
  std::shared_ptr<ast::VisItem> fun;
  std::shared_ptr<ast::Expression> macro;

public:
  ExternalItem(Location loc) : Node(loc){};

  void setOuterAttributes(std::span<OuterAttribute> outer) {
    outerAttributes = {outer.begin(), outer.end()};
  }

  void setStaticItem(std::shared_ptr<ast::VisItem> st) { stat = st; }
  void setFunction(std::shared_ptr<ast::VisItem> fn) { fun = fn; }
  void setMacroInvocation(std::shared_ptr<ast::Expression> mac) { macro = mac; }
};

} // namespace rust_compiler::ast

// FIXME
