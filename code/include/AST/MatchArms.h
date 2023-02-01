#pragma once

#include "AST/AST.h"
#include "AST/MatchArm.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class MatchArms : public Node {
public:
  MatchArms(Location loc) : Node(loc){};
};

} // namespace rust_compiler::ast
