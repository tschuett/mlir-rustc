#pragma once

#include "AST/AST.h"

// https://doc.rust-lang.org/reference/types.html

namespace rust_compiler::ast::types {

// Primitive types

// sequence types

// user-defined types

// function types

// pointer types

// trait types

class Type : public Node {
public:
  Type(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast::types
