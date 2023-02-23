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

  void addParam(std::shared_ptr<Expression> pa) { params.push_back(pa); }
};

} // namespace rust_compiler::ast
