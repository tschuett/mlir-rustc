#pragma once

#include "AST/AST.h"
#include "AST/Function.h"
#include "AST/Item.h"
#include "AST/Visiblity.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

class Module : public Item {
  std::string path;
  std::vector<std::shared_ptr<Item>> items;
  std::vector<std::shared_ptr<Function>> funs;

public:
  Module(rust_compiler::Location location, std::string_view path)
      : Item(location), path(path){};

  void setVisibility(Visibility vis);

  void addItem(std::shared_ptr<Item> item);

  void addFunction(std::shared_ptr<Function> func);

  size_t getTokens() override;

  std::span<std::shared_ptr<Function>> getFuncs();
};

} // namespace rust_compiler::ast
