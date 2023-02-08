#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/AST.h"
#include "AST/OuterAttributes.h"
#include "Location.h"

#include <memory>

namespace rust_compiler::ast {

class VisItem;

enum class ItemKind { VisItem, MacroItem };

class Item : public Node {
  std::shared_ptr<OuterAttributes> outerAttributes;
  std::shared_ptr<VisItem> visItem;

  ItemKind kind;

public:
  explicit Item(rust_compiler::Location location, ItemKind kind)
      : Node(location), kind(kind) {}

  void setOuterAttributes(std::shared_ptr<OuterAttributes> outer);
  void setVisItem(std::shared_ptr<VisItem> visItem);

  std::shared_ptr<OuterAttributes> getOuterAttributes() const;

  std::shared_ptr<VisItem> getVisItem() const;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
