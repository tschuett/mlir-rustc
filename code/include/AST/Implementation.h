#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

class Implementation : public VisItem {
public:
  Implementation(const adt::CanonicalPath &path, Location loc)
      : VisItem(path, loc, VisItemKind::Implementation) {}
};

} // namespace rust_compiler::ast
