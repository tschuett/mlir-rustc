#pragma once

#include "WhereClauseItem.h"

namespace rust_compiler::ast {

class LifetimeWhereClauseItem : public WhereClauseItem {
public:
  LifetimeWhereClauseItem(Location loc)
      : WhereClauseItem(loc, WhereClauseItemKind::LifetimeWhereClauseItem) {}
};

} // namespace rust_compiler::ast
