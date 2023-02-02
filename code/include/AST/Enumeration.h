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
  Enumeration(const adt::CanonicalPath &path, Location loc)
    : VisItem(path, loc, VisItemKind::Enumeration), enumItems(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
