#pragma once

#include "AST/VisItem.h"
#include "AST/InnerAttribute.h"
#include "AST/ExternalItem.h"
#include "AST/Abi.h"

#include <optional>

namespace rust_compiler::ast {

class ExternBlock : public VisItem {
  bool unsafe;
  std::optional<Abi> abi;
  std::vector<InnerAttribute> innerAttributes;
  std::vector<ExternalItem> externalItems;

public:
  ExternBlock(const adt::CanonicalPath &path, Location loc)
    : VisItem(path, loc, VisItemKind::ExternBlock) {}

   size_t getTokens() override;
};

} // namespace rust_compiler::ast
