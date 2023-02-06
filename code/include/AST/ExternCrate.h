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
  ExternCrate(const adt::CanonicalPath &path, Location loc)
      : VisItem(path, loc, VisItemKind::ExternCrate) {}
};

} // namespace rust_compiler::ast
