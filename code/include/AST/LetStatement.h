#pragma once

#include "AST/Statement.h"
#include "AST/VariableDeclaration.h"
#include "AST/Types/Types.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class LetStatement : public Statement {
  VariableDeclaration var;
  std::shared_ptr<Type> type;
public:
};

} // namespace rust_compiler::ast
