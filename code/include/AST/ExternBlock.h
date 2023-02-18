#pragma once

#include "AST/Abi.h"
#include "AST/ExternalItem.h"
#include "AST/InnerAttribute.h"
#include "AST/VisItem.h"

#include <optional>

namespace rust_compiler::ast {

class ExternBlock : public VisItem {
  bool unsafe;
  std::optional<Abi> abi;
  std::vector<InnerAttribute> innerAttributes;
  std::vector<ExternalItem> externalItems;

public:
  ExternBlock(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::ExternBlock, vis) {}

  bool isUnsafe() const { return unsafe; };
};

} // namespace rust_compiler::ast
