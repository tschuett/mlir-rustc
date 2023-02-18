#pragma once

#include "AST/Expression.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Scrutinee.h"
#include "Location.h"

namespace rust_compiler::ast {

class IfLetExpression final : public ExpressionWithBlock {
  std::shared_ptr<ast::patterns::Pattern> pattern;
  Scrutinee scrutinee;
  std::shared_ptr<ast::Expression> block;
  std::shared_ptr<ast::Expression> trailingBlock;
  std::shared_ptr<ast::Expression> trailingIf;
  std::shared_ptr<ast::Expression> trailingIfLet;

public:
  IfLetExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfLetExpression),
        scrutinee(loc) {}

  void setPattern(std::shared_ptr<ast::patterns::Pattern> _pattern) {
    pattern = _pattern;
  }

  void setScrutinee(ast::Scrutinee _scrutinee) { scrutinee = _scrutinee; }

  void setBlock(std::shared_ptr<ast::Expression> _block) { block = _block; }

  void setTailBlock(std::shared_ptr<ast::Expression> block) {
    trailingBlock = block;
  }

  void setIf(std::shared_ptr<ast::Expression> _if) { trailingIf = _if; }

  void setIfLet(std::shared_ptr<ast::Expression> _if) { trailingIfLet = _if; }
};

} // namespace rust_compiler::ast
