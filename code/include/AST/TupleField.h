#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Visiblity.h"

#include <memory>
#include <optional>
#include <vector>
#include <span>

namespace rust_compiler::ast {

class TupleField : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  TupleField(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }

  void setVisibility(const Visibility &vis) { visibility = vis; }

  void setType(std::shared_ptr<ast::types::TypeExpression> t) { type = t; }
};

} // namespace rust_compiler::ast
