#pragma once

#include "AST/Statement.h"


namespace rust_compiler::ast {

class EmptyStatement final : public Statement {

public:
  EmptyStatement(Location loc)
      : Statement(loc, StatementKind::EmptyStatement){};
};

} // namespace rust_compiler::ast
