#pragma once

#include "AST/UseTree.h"
#include "AST/VisItem.h"

#include <memory>

namespace rust_compiler::ast {

class UseDeclaration : public VisItem {
  std::shared_ptr<use_tree::UseTree> tree;

public:
  UseDeclaration(rust_compiler::Location location)
      : VisItem(location, VisItemKind::UseDeclaration){};

  void setComponent(std::shared_ptr<use_tree::UseTree> tree);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
