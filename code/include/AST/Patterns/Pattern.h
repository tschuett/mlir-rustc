#pragma once

#include "AST/AST.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class PatternNoTopAlt;

class Pattern : public Node {
  std::vector<std::shared_ptr<ast::patterns::PatternNoTopAlt>> patterns;
  bool leadingOr = false;

public:
  Pattern(Location loc) : Node(loc) {}

  void setLeadingOr() { leadingOr = true; }

  void addPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat) {
    patterns.push_back(pat);
  }

  std::vector<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  getPatterns() const {
    return patterns;
  }
};

} // namespace rust_compiler::ast::patterns

// https://doc.rust-lang.org/reference/patterns.html
