#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class SelfParam : public Node {

public:
  SelfParam(Location loc) : Node(loc) {}

    size_t getTokens() override;

};

} // namespace rust_compiler::ast
