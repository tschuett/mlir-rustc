#pragma once

#include "AST/Item.h"
#include "AST/Visiblity.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

enum class VisItemKind {
  Module,
  ExternCrate,
  UseDeclaration,
  Function,
  TypeAlias,
  Struct,
  Enumeration,
  Union,
  ConstantItem,
  StaticItem,
  Trait,
  Implementation,
  ExternBlock,
};

class VisItem : public Item {
  VisItemKind kind;
  std::optional<Visibility> vis;

public:
  explicit VisItem(rust_compiler::Location location, VisItemKind kind,
                   std::optional<Visibility> vis)
      : Item(location, ItemKind::VisItem), kind(kind), vis(vis) {}

  VisItemKind getKind() const { return kind; }

  std::optional<Visibility> getVisibility() const { return vis; }
};

} // namespace rust_compiler::ast

// FIXME: incomplete

// InnerAttribute,
//   ClippyAttribute
