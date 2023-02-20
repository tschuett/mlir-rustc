#pragma once

#include "AST/Abi.h"
#include "AST/ExternalItem.h"
#include "AST/InnerAttribute.h"
#include "AST/VisItem.h"

#include <optional>
#include <vector>

namespace rust_compiler::ast {

class ExternBlock : public VisItem {
  bool unsafe = false;
  std::optional<Abi> abi;
  std::vector<InnerAttribute> innerAttributes;
  std::vector<ExternalItem> externalItems;

public:
  ExternBlock(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::ExternBlock, vis) {}

  bool isUnsafe() const { return unsafe; };
  void setUnsafe() { unsafe = true; }

  void setAbi(const Abi &ab) { abi = ab; }

  void setInnerAttributes(std::span<InnerAttribute> inner) {
    innerAttributes = {inner.begin(), inner.end()};
  }

  void addItem(const ExternalItem &ex) { externalItems.push_back(ex); }
};

} // namespace rust_compiler::ast
