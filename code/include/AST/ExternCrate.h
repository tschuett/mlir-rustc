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
  ExternCrate(Location loc)
    : VisItem(loc, VisItemKind::ExternCrate), crateRef(loc) {}

   size_t getTokens() override;
};

} // namespace rust_compiler::ast
