#pragma once

#include "AST/VisItem.h"

#include "AST/CrateRef.h"
#include "AST/AsClause.h"

#include <optional>

namespace rust_compiler::ast {

class ExternCrate : public VisItem {
  CrateRef crateRef;
  std::optional<AsClause> asClasue;

public:
  ExternCrate(Location loc, std::optional<Visibility> vis)
    : VisItem(loc, VisItemKind::ExternCrate, vis), crateRef(loc) {}
};

} // namespace rust_compiler::ast
