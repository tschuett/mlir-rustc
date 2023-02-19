#pragma once

#include "WhereClauseItem.h"

namespace rust_compiler::ast {

class LifetimeWhereClauseItem : public WhereClauseItem {
  Lifetime lifetime;
  LifetimeBounds bounds;

public:
  LifetimeWhereClauseItem(Location loc)
      : WhereClauseItem(loc, WhereClauseItemKind::LifetimeWhereClauseItem),
        lifetime(loc), bounds(loc) {}

  void setForLifetimes(const ast::Lifetime &lt) { lifetime = lt; }

  void setLifetimeBounds(const ast::LifetimeBounds &lb) { bounds = lb; }
};

} // namespace rust_compiler::ast
