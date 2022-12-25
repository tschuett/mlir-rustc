#pragma once

#include "AST/Item.h"

namespace rust_compiler::ast {

class OuterAttribute : public Item {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
