#pragma once

#include "AST/AST.h"
#include "AST/Attr.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

class InnerAttribute : public Node {
  Attr attr;

public:
  InnerAttribute(rust_compiler::Location location)
      : Node(location), attr(location) {}

  void setAttr(const Attr &at) { attr = at; }
};

} // namespace rust_compiler::ast
