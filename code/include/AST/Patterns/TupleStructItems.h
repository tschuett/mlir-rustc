#pragma once

#include "AST/AST.h"
#include "AST/Patterns/Pattern.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class TupleStructItems : public Node {
  bool trailingComma;
  std::vector<std::shared_ptr<ast::patterns::Pattern>> patterns;

public:
  TupleStructItems(Location loc) : Node(loc) {}

  void addPattern(std::shared_ptr<ast::patterns::Pattern> pat) {
    patterns.push_back(pat);
  }
  void setTrailingComma() { trailingComma = true; }
};

} // namespace rust_compiler::ast::patterns
