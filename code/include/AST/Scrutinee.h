#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class Scrutinee : public Node {
  std::shared_ptr<ast::Expression> expr;

public:
  Scrutinee(Location loc) : Node(loc){};

  size_t getTokens() override;
};

} // namespace rust_compiler::ast

// FIXME:
