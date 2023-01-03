#pragma once

#include <AST/AST.h>

namespace rust_compiler::ast {

class Visibility : public Node {
public:
  Visibility();

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
