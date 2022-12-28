#pragma once

#include "AST/AST.h"
#include "AST/FunctionSignature.h"

namespace rust_compiler::ast {

class Function : public Node {

public:
  FunctionSignature getSignature();
};

} // namespace rust_compiler::ast
