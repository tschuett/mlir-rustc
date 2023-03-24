#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

enum StructKind { StructStruct2, TupleStruct2 };

class Struct : public VisItem {
  StructKind kind;

public:
  Struct(Location loc, StructKind kind, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Struct, vis), kind(kind) {}

  StructKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
