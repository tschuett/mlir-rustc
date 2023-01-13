#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

namespace rust_compiler::ast {

class OuterAttributes : public Node {
  std::vector<std::shared_ptr<OuterAttribute>> outerAttributes;

public:
  OuterAttributes(Location loc) : Node(loc) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
