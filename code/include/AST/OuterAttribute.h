#pragma once

#include "AST/AST.h"
#include "AST/AttrInput.h"
#include "AST/SimplePath.h"

#include <optional>

namespace rust_compiler::ast {

class OuterAttribute : public Node {
  SimplePath path;
  std::optional<AttrInput> attrInput;

public:
  OuterAttribute(Location loc) : Node(loc), path(loc) {}

  SimplePath getPath() const;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
