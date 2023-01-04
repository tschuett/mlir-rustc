#pragma once

#include "AST/Item.h"
#include "AST/UseTree.h"

#include <memory>

namespace rust_compiler::ast {

class UseDeclaration : public Item {
  std::shared_ptr<use_tree::UseTree> tree;

public:
  UseDeclaration(rust_compiler::Location location) : Item(location){};

  void setComponent(std::shared_ptr<use_tree::UseTree> tree);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
