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

enum class ModuleKind { Module, ModuleTree };

class Module : public Item {
  ModuleKind kind;
  std::string path;
  std::vector<std::shared_ptr<Item>> items;
  std::vector<std::shared_ptr<Function>> funs;
  Visibility vis = VisibilityKind::Private;
public:
  Module(rust_compiler::Location location, ModuleKind kind, std::string_view path)
      : Item(location), kind(kind), path(path){};

  void setVisibility(Visibility vis);

  void addItem(std::shared_ptr<Item> item);

  void addFunction(std::shared_ptr<Function> func);

  size_t getTokens() override;

  std::span<std::shared_ptr<Function>> getFuncs();
};

} // namespace rust_compiler::ast
