#pragma once

#include "AST/Item.h"

namespace rust_compiler::ast {

class InnerAttribute : public Item {
public:
  InnerAttribute(mlir::Location location) : Item(location) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
