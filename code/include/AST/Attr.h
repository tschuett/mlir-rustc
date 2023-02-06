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

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
