#include "AST/BreakExpression.h"
#include "BlockExpressionVisitor.h"
#include "Sema/TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

class BreakExpressionCollector : public BlockExpressionVisitor {
  std::vector<std::shared_ptr<ast::BreakExpression>> breaks;

public:
  virtual void
  visitBreakExpression(std::shared_ptr<ast::Expression> breakExpr) override {
    breaks.push_back(std::static_pointer_cast<BreakExpression>(breakExpr));
  }

  std::vector<std::shared_ptr<ast::BreakExpression>> getBreaks() const {
    return breaks;
  };
};

void TypeChecking::checkInfiniteLoopExpression(
    std::shared_ptr<ast::InfiniteLoopExpression> loop) {

  BreakExpressionCollector collector;

  run(loop->getBody(), &collector);

  std::vector<std::shared_ptr<ast::BreakExpression>> breaks =
      collector.getBreaks();
  // collect breaks in BlockExpression
}

} // namespace rust_compiler::sema

// Unit or breaks
