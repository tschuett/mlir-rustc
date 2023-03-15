#pragma once

#include "AST/Expression.h"
#include "AST/Statement.h"

#include <memory>
#include <variant>

namespace rust_compiler::parser {

// https://doc.rust-lang.org/reference/expressions/block-expr.html

/// Tool for parsing BlockExpressions
class ExpressionOrStatement {
  std::variant<std::shared_ptr<ast::Expression>,
               std::shared_ptr<ast::Statement>, std::shared_ptr<ast::Item>>
      data;

  // std::shared_ptr<ast::Expression> expr;
  // std::shared_ptr<ast::Statement> stmt;

public:
  ExpressionOrStatement(std::shared_ptr<ast::Expression> e) { data = e; }
  ExpressionOrStatement(std::shared_ptr<ast::Statement> s) { data = s; }
  ExpressionOrStatement(std::shared_ptr<ast::Item> i) { data = i; }
};

} // namespace rust_compiler::parser
