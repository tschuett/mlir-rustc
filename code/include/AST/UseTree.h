#pragma once

#include "AST/AST.h"

#include <cstddef>

namespace rust_compiler::ast {

class UseTree : public Node {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
