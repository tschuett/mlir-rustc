#pragma once

#include "AST/BlockExpression.h"
#include "AST/Expression.h"
#include "AST/LoopExpression.h"
#include "AST/Patterns/Patterns.h"
#include "AST/Scrutinee.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class PredicatePatternLoopExpression final : public LoopExpression {
  std::shared_ptr<ast::Expression> body;
  std::shared_ptr<patterns::Pattern> pattern;
  Scrutinee scrutinee;
  std::optional<std::string> loopLabel;

public:
  PredicatePatternLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::PredicatePatternLoopExpression),
        scrutinee(loc) {}

  //  void setCondition(std::shared_ptr<ast::Expression>);
  //  void setBody(std::shared_ptr<ast::BlockExpression>);
  //
  //  std::shared_ptr<ast::Expression> getCondition() const;
  //  std::shared_ptr<ast::BlockExpression> getBody() const;
  //

  void setPattern(std::shared_ptr<patterns::Pattern> pat) { pattern = pat; }
  void setScrutinee(const Scrutinee &expr) { scrutinee = expr; }
  void setBody(std::shared_ptr<Expression> bod) { body = bod; }

  void setLabel(std::string_view lab) { loopLabel = lab; }
};

} // namespace rust_compiler::ast
