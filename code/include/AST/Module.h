#pragma once

#include "AST/AST.h"
#include "AST/Item.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast {

class Module : public Node {
  std::string path;
  // std::vector<Item> items;

public:
  Module(std::string_view path) : path(path){};

  void addItem(std::shared_ptr<Item> &item);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
