#pragma once

#include "AST/AST.h"
#include "AST/Function.h"
#include "AST/Item.h"
#include "AST/VisItem.h"
#include "AST/Visiblity.h"

#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

enum class ModuleKind { Module, ModuleTree };

class Module : public VisItem {
  Visibility vis;
  ModuleKind kind;
  std::vector<std::shared_ptr<Item>> items;
  // std::vector<std::shared_ptr<Function>> funs;

public:
  Module(const adt::CanonicalPath &path, rust_compiler::Location loc,
         ModuleKind kind)
      : VisItem(path, loc, VisItemKind::Module),
        vis(loc, VisibilityKind::Private), kind(kind){};

  ModuleKind getModuleKind() const { return kind; }
  void setVisibility(Visibility vis);

  void addItem(std::shared_ptr<Item> item);

  // void addFunction(std::shared_ptr<Function> func);

  size_t getTokens() override;

  std::span<std::shared_ptr<Item>> getItems();
};

} // namespace rust_compiler::ast
