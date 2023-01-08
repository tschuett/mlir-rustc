#pragma once

#include "AST/AST.h"
#include "Location.h"

namespace rust_compiler::ast {

class Item : public Node {

public:
  explicit Item(rust_compiler::Location location) : Node{location} {}
};

} // namespace rust_compiler::ast
