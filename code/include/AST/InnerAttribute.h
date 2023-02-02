#pragma once

#include "AST/AST.h"
#include "AST/AttrInput.h"
#include "AST/SimplePath.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

class InnerAttribute : public Node {
  SimplePath path;
  std::optional<AttrInput> attrInput;

public:
  InnerAttribute(rust_compiler::Location location)
      : Node(location), path(location) {}

  SimplePath getPath() const;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
