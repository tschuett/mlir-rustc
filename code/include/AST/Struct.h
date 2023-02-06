#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

enum StructKind { StructStruct, TupleStruct };

class Struct : public VisItem {
  StructKind kind;

public:
  Struct(const adt::CanonicalPath &path, Location loc, StructKind kind)
      : VisItem(path, loc, VisItemKind::Struct), kind(kind) {}
};

} // namespace rust_compiler::ast
