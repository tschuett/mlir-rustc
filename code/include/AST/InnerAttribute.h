#pragma once

#include "AST/Item.h"
#include "Location.h"

namespace rust_compiler::ast {

class InnerAttribute : public Item {
public:
  InnerAttribute(rust_compiler::Location location) : Item(location) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
