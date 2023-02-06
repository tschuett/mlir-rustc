#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

class AssociatedItem : public VisItem {

public:
  AssociatedItem(const adt::CanonicalPath &path, Location loc)
      : VisItem(path, loc, VisItemKind::AssociatedItem) {}
};

} // namespace rust_compiler::ast
