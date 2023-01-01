#pragma once

#include "AST/Statement.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class Item : public Statement {

public:
  explicit Item(mlir::Location location) : Statement{location} {}
};

} // namespace rust_compiler::ast
