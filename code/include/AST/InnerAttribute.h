#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

class InnerAttribute : public Node {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
