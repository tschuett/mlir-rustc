#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class CallParams : public Node {
  std::vector<std::shared_ptr<Expression>> params;

public:
  CallParams(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
