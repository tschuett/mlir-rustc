#pragma once

#include "AST/AST.h"
#include "AST/GenericParam.h"

#include <vector>

namespace rust_compiler::ast {

class GenericParams : public Node {
  bool trailingComma = false;
  std::vector<GenericParam> genericParams;

public:
  GenericParams(Location loc) : Node(loc){};

  void addGenericParam(const GenericParam &gp) { genericParams.push_back(gp); }

  void setTrailingComma() { trailingComma = true; }

  bool hasParams() const { return !genericParams.empty(); }

  std::vector<GenericParam> getGenericParams() const { return genericParams; }
};

} // namespace rust_compiler::ast
