#pragma once

#include "AST/AST.h"
#include "AST/Function.h"
#include "AST/InnerAttribute.h"
#include "AST/Item.h"
#include "AST/VisItem.h"
#include "AST/Visiblity.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

enum class ModuleKind { Module, ModuleTree };

class Module : public VisItem {
  ModuleKind kind;
  bool unsafe = false;
  ;
  std::string identifier;
  std::vector<InnerAttribute> innerAttributes;
  std::vector<std::shared_ptr<Item>> items;

public:
  Module(rust_compiler::Location loc, std::optional<Visibility> vis,
         ModuleKind kind, std::string_view modName)
      : VisItem(loc, VisItemKind::Module, vis), kind(kind){};

  ModuleKind getModuleKind() const { return kind; }
  std::string_view getModuleName() const { return identifier; }

  void addItem(std::shared_ptr<Item> item);

  // void addFunction(std::shared_ptr<Function> func);

  std::span<std::shared_ptr<Item>> getItems();
  void setItems(std::span<std::shared_ptr<Item>> items);

  void setInnerAttributes(std::span<ast::InnerAttribute> in) {
    innerAttributes = {in.begin(), in.end()};
  }
  void setItem(std::span<std::shared_ptr<ast::Item>> it) {
    items = {it.begin(), it.end()};
  }
  void setUnsafe() { unsafe = true; }
};

} // namespace rust_compiler::ast
