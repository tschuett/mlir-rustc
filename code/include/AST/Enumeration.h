#pragma once

#include "AST/GenericParams.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "AST/EnumItems.h"

#include <string>

namespace rust_compiler::ast {

class Enumeration : public VisItem {
  std::string identifier;
  std::shared_ptr<GenericParams> genericParams;
  std::shared_ptr<WhereClause> whereClause;
  EnumItems enumItems;

public:
  Enumeration(Location loc, std::optional<Visibility> vis)
    : VisItem(loc, VisItemKind::Enumeration, vis), enumItems(loc) {}
};

} // namespace rust_compiler::ast
