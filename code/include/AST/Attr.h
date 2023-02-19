#pragma once

#include "AST/AttrInput.h"
#include "AST/SimplePath.h"

#include <optional>

namespace rust_compiler::ast {

class Attr : public Node {
  SimplePath path;
  std::optional<AttrInput> attrInput;

public:
  Attr(Location loc) : Node(loc), path(loc) {}

  void setSimplePath(const SimplePath &sim) { path = sim; }
  void setAttrInput(const AttrInput &input) { attrInput = input; }
};

} // namespace rust_compiler::ast
