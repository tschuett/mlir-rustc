#pragma once

#include "AST/AST.h"
#include "AST/Types/Types.h"

#include <vector>
#include <memory>

namespace rust_compiler::ast::types {

class TypePathFnInputs final : public Node {
  std::vector<std::shared_ptr<types::Type>> types;
  bool trailingcomma;

public:
  TypePathFnInputs(Location loc) : Node(loc) {}

   size_t getTokens() override;
};

} // namespace rust_compiler::ast::types
