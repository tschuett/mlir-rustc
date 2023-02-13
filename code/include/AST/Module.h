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
  Visibility vis;
  ModuleKind kind;
  bool unsafe;
  std::string identifier;
  std::vector<std::shared_ptr<InnerAttribute>> innerAttributes;
  std::vector<std::shared_ptr<Item>> items;

public:
  Module(rust_compiler::Location loc, ModuleKind kind)
      : VisItem(loc, VisItemKind::Module), vis(loc, VisibilityKind::Private),
        kind(kind){};

  ModuleKind getModuleKind() const { return kind; }
  std::string_view getModuleName() const { return identifier; }
  void setVisibility(Visibility vis);

  void addItem(std::shared_ptr<Item> item);

  // void addFunction(std::shared_ptr<Function> func);

  size_t getTokens() override;

  std::span<std::shared_ptr<Item>> getItems();
  void setItems(std::span<std::shared_ptr<Item>> items);

};

} // namespace rust_compiler::ast
