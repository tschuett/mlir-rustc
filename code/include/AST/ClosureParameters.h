#pragma once

#include "AST/AST.h"
#include "AST/ClosureParam.h"

namespace rust_compiler::ast {

class ClosureParameters : public Node {
  std::vector<ClosureParam> parameters;
  bool trailingComma;

public:
  ClosureParameters(Location loc) : Node(loc) {}

  bool isTrailingComma() const { return trailingComma; }

  void addParam(const ClosureParam &cp) { parameters.push_back(cp); }

  std::vector<ClosureParam> getParameters() const { return parameters; }
};

} // namespace rust_compiler::ast
