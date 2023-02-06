#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class CrateRef : public Node {

public:
  CrateRef(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
