#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

enum class ImplementationKind { InherentImpl, TraitImpl };

class Implementation : public VisItem {
  ImplementationKind kind;

public:
  Implementation(const adt::CanonicalPath &path, ImplementationKind kind,
                 Location loc)
      : VisItem(path, loc, VisItemKind::Implementation), kind(kind) {}
};

} // namespace rust_compiler::ast
