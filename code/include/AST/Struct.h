#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

enum StructKind { StructStruct, TupleStruct };

class Struct : public VisItem {
  StructKind kind;

public:
  Struct(Location loc, StructKind kind)
      : VisItem(loc, VisItemKind::Struct), kind(kind) {}
};

} // namespace rust_compiler::ast
