#pragma once

#include "AST/AST.h"
#include "AST/GenericParams.h"

#include <vector>

namespace rust_compiler::ast::types {

class ForLifetimes : public Node {
  GenericParams genericParams;

public:
  ForLifetimes(Location loc) : Node(loc), genericParams(loc) {}

  void setGenericParams(const GenericParams &par) { genericParams = par; }
};

} // namespace rust_compiler::ast::types
