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
  ExternBlock
};

class VisItem : public Item {
  VisItemKind kind;
  std::optional<Visibility> vis;

public:
  explicit VisItem(const CanonicalPath &path, rust_compiler::Location location,
                   VisItemKind kind)
      : Item(path, location), kind(kind) {}

  VisItemKind getKind() const { return kind; }

  //  size_t getTokens() override;
};

} // namespace rust_compiler::ast

// FIXME: incomplete

// InnerAttribute,
//   ClippyAttribute
