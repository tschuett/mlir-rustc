#pragma once

#include "AST/Expression.h"
#include "Location.h"

namespace rust_compiler::ast {

class QualifiedInPathType final : public Node {

public:
  QualifiedInPathType(Location loc) : Node(loc) {}

   size_t getTokens() override;
};

} // namespace rust_compiler::ast
