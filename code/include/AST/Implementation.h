#pragma once

#include "AST/VisItem.h"

namespace rust_compiler::ast {

enum class ImplementationKind { InherentImpl, TraitImpl };

class Implementation : public VisItem {
  ImplementationKind kind;

public:
  Implementation(ImplementationKind kind,
                 Location loc, std::optional<Visibility> vis)
    : VisItem(loc, VisItemKind::Implementation, vis), kind(kind) {}

  ImplementationKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
