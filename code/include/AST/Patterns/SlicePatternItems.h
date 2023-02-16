#pragma once

#include "AST/AST.h"
#include "AST/Patterns/Patterns.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class SlicePatternItems : public Node {
  std::vector<std::shared_ptr<ast::patterns::Pattern>> patterns;
  bool trailingComa = false;

public:
  SlicePatternItems(Location loc) : Node(loc) {}

  bool hasTrailingComma();
};

} // namespace rust_compiler::ast::patterns
