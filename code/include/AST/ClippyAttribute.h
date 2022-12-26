#pragma once

#include "AST/Item.h"

namespace rust_compiler::ast {

class ClippyAttribute : public Item {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
