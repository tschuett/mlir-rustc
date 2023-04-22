#pragma once

#include "AST/Expression.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Scrutinee.h"
#include "Location.h"

namespace rust_compiler::ast {

enum struct IfLetExpressionKind { NoElse, ElseBlock, ElseIf, ElseIfLet };

class IfLetExpression final : public ExpressionWithBlock {
  std::shared_ptr<ast::patterns::Pattern> pattern;
  Scrutinee scrutinee;
  std::shared_ptr<ast::Expression> block;
  std::shared_ptr<ast::Expression> trailingBlock;
  std::shared_ptr<ast::Expression> trailingIf;
  std::shared_ptr<ast::Expression> trailingIfLet;
  IfLetExpressionKind kind = IfLetExpressionKind::NoElse;

public:
  IfLetExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfLetExpression),
        scrutinee(loc) {}

  void setPattern(std::shared_ptr<ast::patterns::Pattern> _pattern) {
    pattern = _pattern;
  }

  IfLetExpressionKind getKind() const { return kind; }

  void setScrutinee(ast::Scrutinee _scrutinee) { scrutinee = _scrutinee; }

  void setBlock(std::shared_ptr<ast::Expression> _block) { block = _block; }

  void setTailBlock(std::shared_ptr<ast::Expression> block) {
    trailingBlock = block;
    kind = IfLetExpressionKind::ElseBlock;
  }

  std::shared_ptr<ast::Expression> getTailBlock() const { return block; }

  void setIf(std::shared_ptr<ast::Expression> _if) {
    trailingIf = _if;
    kind = IfLetExpressionKind::ElseIf;
  }

  std::shared_ptr<ast::Expression> getIf() const { return trailingIf; }

  void setIfLet(std::shared_ptr<ast::Expression> _if) {
    trailingIfLet = _if;
    kind = IfLetExpressionKind::ElseIfLet;
  }

  std::shared_ptr<ast::Expression> getIfLet() const { return trailingIfLet; }

  std::shared_ptr<ast::Expression> getBlock() const { return block; }

  Scrutinee getScrutinee() const { return scrutinee; }

  std::shared_ptr<ast::patterns::Pattern> getPatterns() const {
    return pattern;
  }
};

} // namespace rust_compiler::ast
