#pragma once

#include "AST/Expression.h"
#include "AST/LoopExpression.h"
#include "AST/Patterns/Pattern.h"
#include "AST/Scrutinee.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class PredicatePatternLoopExpression final : public LoopExpression {
  std::shared_ptr<ast::Expression> body;
  std::shared_ptr<ast::patterns::Pattern> pattern;
  Scrutinee scrutinee;
  std::optional<std::string> loopLabel;

public:
  PredicatePatternLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::PredicatePatternLoopExpression),
        scrutinee(loc) {}

  void setPattern(std::shared_ptr<ast::patterns::Pattern> pat) { pattern = pat; }
  void setScrutinee(const Scrutinee &expr) { scrutinee = expr; }
  void setBody(std::shared_ptr<Expression> bod) { body = bod; }

  void setLabel(std::string_view lab) { loopLabel = lab; }

  Scrutinee &getScrutinee()  { return scrutinee; }
  std::shared_ptr<ast::Expression> getBody() const { return body; }
};

} // namespace rust_compiler::ast
