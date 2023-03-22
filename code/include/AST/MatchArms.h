#pragma once

#include "AST/AST.h"
#include "AST/MatchArm.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class MatchArms : public Node {
  std::vector<std::pair<MatchArm, std::shared_ptr<Expression>>> arms;

public:
  MatchArms(Location loc) : Node(loc){};

  void addArm(const MatchArm &arm, std::shared_ptr<Expression> e) {
    arms.push_back({arm, e});
  }

  std::vector<std::pair<MatchArm, std::shared_ptr<Expression>>>
  getArms() const {
    return arms;
  }
};

} // namespace rust_compiler::ast
