#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

enum class ImplementationKind { InherentImpl, TraitImpl };

class Implementation : public VisItem {
  ImplementationKind kind;

public:
  Implementation(ImplementationKind kind,
                 Location loc)
      : VisItem(loc, VisItemKind::Implementation), kind(kind) {}

  ImplementationKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
