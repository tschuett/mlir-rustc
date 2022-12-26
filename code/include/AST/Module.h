#pragma once

#include "AST/AST.h"
#include "AST/Item.h"
#include "AST/Function.h"

#include <string>
#include <string_view>
#include <vector>
#include <span>

namespace rust_compiler::ast {

class Module : public Node {
  std::string path;
  // std::vector<Item> items;
  std::vector<std::shared_ptr<Function>> funs;

public:
  Module(std::string_view path) : path(path){};

  void addItem(std::shared_ptr<Item> item);

  void addFunction(std::shared_ptr<Function> func);

  size_t getTokens() override;

  std::span<std::shared_ptr<Function>> getFuncs();

};

} // namespace rust_compiler::ast
