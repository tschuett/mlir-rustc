#pragma once

#include "AST/Statement.h"
#include "Location.h"

namespace rust_compiler::ast {

class Item : public Statement {

public:
  explicit Item(rust_compiler::Location location) : Statement{location} {}
};

} // namespace rust_compiler::ast
