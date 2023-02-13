#pragma once

#include "AST/VisItem.h"

#include <memory>

namespace rust_compiler::ast::use_tree {
class UseTree;
}

namespace rust_compiler::ast {

class UseDeclaration : public VisItem {
  std::shared_ptr<use_tree::UseTree> tree;

public:
  UseDeclaration(rust_compiler::Location location,
                 std::optional<Visibility> vis)
      : VisItem(location, VisItemKind::UseDeclaration, vis){};

  void setComponent(std::shared_ptr<use_tree::UseTree> tree);
};

} // namespace rust_compiler::ast
