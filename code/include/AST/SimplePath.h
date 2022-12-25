#pragma once

#include "AST/AST.h"

#include <cstddef>

namespace rust_compiler::ast {

class SimplePath : public Node {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
