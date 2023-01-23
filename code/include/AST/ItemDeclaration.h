#pragma once

#include "AST/AST.h"
#include "AST/Item.h"
#include "AST/Statement.h"
#include "Location.h"

namespace rust_compiler::ast {

class ItemDeclaration : public Statement {
  std::shared_ptr<Item> item;

public:
  ItemDeclaration(Location loc) : Statement(loc, StatementKind::ItemDeclaration) {}
};

} // namespace rust_compiler::ast
