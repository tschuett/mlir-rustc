#pragma once

#include "AST/AST.h"

#include "AST/ClosureParam.h"

namespace rust_compiler::ast {

class ClosureParameters : public Node {
  std::vector<ClosureParam> parameters;
  bool trailingComma;
public:
  ClosureParameters(Location loc) : Node(loc) {}

   size_t getTokens() override;
};

} // namespace rust_compiler::ast
