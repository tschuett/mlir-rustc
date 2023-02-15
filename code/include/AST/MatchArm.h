#pragma once

#include "AST/AST.h"
#include "AST/MatchArmGuard.h"
#include "AST/OuterAttribute.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class MatchArm : public Node {
  std::vector<OuterAttribute> outerAttributes;

public:
  MatchArm(Location loc) : Node(loc){};
};

} // namespace rust_compiler::ast
