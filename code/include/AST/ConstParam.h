#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class ConstParam : public Node {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
