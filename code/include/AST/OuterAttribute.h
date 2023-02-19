#pragma once

#include "AST/AST.h"
#include "AST/Attr.h"

#include <optional>

namespace rust_compiler::ast {

class OuterAttribute : public Node {
  Attr attr;

public:
  OuterAttribute(Location loc) : Node(loc), attr(loc) {}

  void setAttr(const Attr &at) { attr = at; }
};

} // namespace rust_compiler::ast
