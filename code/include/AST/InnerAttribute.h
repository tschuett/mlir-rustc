#pragma once

#include "AST/AST.h"
#include "Location.h"

namespace rust_compiler::ast {

class InnerAttribute : public Node {
public:
  InnerAttribute(rust_compiler::Location location) : Node(location) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
