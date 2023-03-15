#pragma once

#include "AST/Expression.h"
#include "AST/Statement.h"

#include <memory>
#include <variant>

namespace rust_compiler::parser {

// https://doc.rust-lang.org/reference/expressions/block-expr.html

enum class ExpressionOrStatementKind { Expression, Statement, Item };

/// Tool for parsing BlockExpressions
class ExpressionOrStatement {
  std::variant<std::shared_ptr<ast::Expression>,
               std::shared_ptr<ast::Statement>, std::shared_ptr<ast::Item>>
      data;

  ExpressionOrStatementKind kind;

public:
  ExpressionOrStatement(std::shared_ptr<ast::Expression> e) {
    data = e;
    kind = ExpressionOrStatementKind::Expression;
  }
  ExpressionOrStatement(std::shared_ptr<ast::Statement> s) {
    data = s;
    kind = ExpressionOrStatementKind::Statement;
  }
  ExpressionOrStatement(std::shared_ptr<ast::Item> i) {
    data = i;
    kind = ExpressionOrStatementKind::Item;
  }

  ExpressionOrStatementKind getKind() const { return kind; }

  std::shared_ptr<ast::Statement> getStatement() const {
    return std::get<std::shared_ptr<ast::Statement>>(data);
  }

  std::shared_ptr<ast::Expression> getExpression() const {
    return std::get<std::shared_ptr<ast::Expression>>(data);
  }
};

} // namespace rust_compiler::parser
