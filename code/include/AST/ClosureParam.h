#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Types/TypeExpression.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class ClosureParam : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<patterns::PatternNoTopAlt> pattern;
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  ClosureParam(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }

  void setPattern(std::shared_ptr<patterns::PatternNoTopAlt> pat) {
    pattern = pat;
  }

  void setType(std::shared_ptr<types::TypeExpression> t) { type = t; }
};

} // namespace rust_compiler::ast
