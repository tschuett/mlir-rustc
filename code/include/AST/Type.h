#pragma once

#include "AST/AST.h"
#include "Location.h"

namespace rust_compiler::ast {

class Type : public Node {

public:
  Type(Location loc) : Node(loc) {}

  // size_t getTokens() override;
};

} // namespace rust_compiler::ast
