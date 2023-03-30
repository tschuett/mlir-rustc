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

  void parseToMetaItem();

  Attr &getAttr() const;
};

} // namespace rust_compiler::ast
