#pragma once

#include "AST/AST.h"
#include "AST/Item.h"

namespace rust_compiler::ast {

class Module : public Node {
  std::string path;
  // std::vector<Item> items;

public:
  Module(std::string_view path) : path(path){};

  void addItem(std::shared_ptr<Item>& item);
};

} // namespace rust_compiler::ast
