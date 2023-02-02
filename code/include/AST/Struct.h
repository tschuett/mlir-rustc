#pragma once

#include "AST/Decls.h"
#include "AST/Item.h"

namespace rust_compiler::ast {

enum StructKind { StructStruct, TupleStruct };

class Struct : public Item {
  StructKind kind;

public:
  Struct(Location loc, StructKind kind) : Item(loc), kind(kind) {}
};

} // namespace rust_compiler::ast
