#pragma once

#include "AST/AST.h"
#include "AST/Patterns/Pattern.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class TuplePatternItems : public Node {
  bool trailingComma;
  bool restPattern;
  std::vector<std::shared_ptr<ast::patterns::Pattern>> patterns;

public:
  TuplePatternItems(Location loc) : Node(loc) {}

  void addPattern(std::shared_ptr<ast::patterns::Pattern> pat) {
    patterns.push_back(pat);
  }
  void setTrailingComma() { trailingComma = true; }

  void setRestPattern() { restPattern = true; }

  std::vector<std::shared_ptr<ast::patterns::Pattern>> getPatterns() const {
    return patterns;
  }
};

} // namespace rust_compiler::ast::patterns
