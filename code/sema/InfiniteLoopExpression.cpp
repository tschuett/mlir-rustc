#include "AST/BreakExpression.h"
#include "BlockExpressionVisitor.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

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

void Sema::analyzeInfiniteLoopExpression(ast::InfiniteLoopExpression *loop) {

  BreakExpressionCollector collector;

  // run(loop->getBody(), &collector);

  std::vector<std::shared_ptr<ast::BreakExpression>> breaks =
      collector.getBreaks();
  // collect breaks in BlockExpression

  // NodeId nodeId = getNodeId(loop);
  //   if (breaks.size() == 0)
  //     typeChecking.isKnownType(
  //         astId, std::make_shared<PrimitiveType>(loop->getLocation(),
  //                                                PrimitiveTypeKind::Unit));
}

} // namespace rust_compiler::sema

// Unit or breaks
