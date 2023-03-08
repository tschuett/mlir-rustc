#pragma once

#include "AST/AST.h"
#include "AST/WhereClauseItem.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class WhereClause : public Node {
  std::vector<std::shared_ptr<WhereClauseItem>> items;
  bool trailingComma = false;

public:
  explicit WhereClause(Location loc) : Node(loc){};

  void addWhereClauseItem(std::shared_ptr<WhereClauseItem> it) {
    items.push_back(it);
  }

  bool hasTrailingComma() const { return trailingComma; }
  void setTrailingComma() { trailingComma = true; }

  size_t getSize() const { return items.size(); }
};

} // namespace rust_compiler::ast
