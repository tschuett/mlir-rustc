#pragma once

#include "AST/AST.h"
#include "AST/MatchArmGuard.h"
#include "AST/OuterAttribute.h"
#include "AST/Patterns/Pattern.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class MatchArm : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<ast::patterns::Pattern> pattern;
  std::optional<MatchArmGuard> guard;

public:
  MatchArm(Location loc) : Node(loc){};

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }

  void setPattern(std::shared_ptr<ast::patterns::Pattern> pat) {
    pattern = pat;
  }

  void setGuard(const MatchArmGuard &ar) { guard = ar; }

  bool hasGuard() const { return guard.has_value(); }

  MatchArmGuard getGuard() const { return *guard; }
};

} // namespace rust_compiler::ast
