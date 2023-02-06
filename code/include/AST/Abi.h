#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class Abi : public Node {

public:
  Abi(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
