#pragma once

#include "AST/UseTree.h"
#include "AST/VisItem.h"

#include <memory>

namespace rust_compiler::ast {

class UseDeclaration : public VisItem {
  use_tree::UseTree tree;

public:
  UseDeclaration(rust_compiler::Location location,
                 std::optional<Visibility> vis)
      : VisItem(location, VisItemKind::UseDeclaration, vis), tree{location} {};

  void setTree(const use_tree::UseTree &tre) { tree = tre; }
};

} // namespace rust_compiler::ast
