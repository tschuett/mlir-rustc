#pragma once

#include "AST/BlockExpression.h"
#include "AST/Expression.h"
#include "AST/Item.h"
#include "AST/ItemDeclaration.h"
#include "AST/Statement.h"
#include "AST/Statements.h"

#include <memory>

namespace rust_compiler::sema {

class BlockExpressionVisitor {
public:
  virtual ~BlockExpressionVisitor() = default;

  virtual void visitStatements(ast::Statements stmts) {}
  virtual void visitStatement(std::shared_ptr<ast::Statement> stmt) {}
  virtual void visitExpressionWithoutBlock(
      std::shared_ptr<ast::ExpressionWithoutBlock> woBlock) {}
  virtual void visitExpressionWithBlock(
      std::shared_ptr<ast::ExpressionWithBlock> withBlock) {}
  virtual void visitItem(std::shared_ptr<ast::Item> let) {}
  virtual void visitItemDeclaration(std::shared_ptr<ast::ItemDeclaration> let) {}
  virtual void visitLetStatement(std::shared_ptr<ast::Statement> let) {}
  virtual void visitExpressionStatement(std::shared_ptr<ast::Statement> let) {}
  virtual void visitMacroInvocationSemi(std::shared_ptr<ast::Statement> let) {}

  virtual void visitBlockExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitUnsafeExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitIfExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitIfLetExpression(std::shared_ptr<ast::Expression> let) {}

  virtual void visitExpression(std::shared_ptr<ast::Expression> let) {}

  virtual void visitLiteralExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitPathExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitOperatorExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitGroupedExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitArrayExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitAwaitExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitIndexExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitTupleExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitTupleIndexingExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitStructExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitCallExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitMethodCallExpression(std::shared_ptr<ast::Expression> let) {
  }
  virtual void visitFieldExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitClosureExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitAsyncBlockExpression(std::shared_ptr<ast::Expression> let) {
  }
  virtual void visitContinueExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitBreakExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitRangeExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitReturnExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitUnderScoreExpression(std::shared_ptr<ast::Expression> let) {
  }
  virtual void visitMacroInvocation(std::shared_ptr<ast::Expression> let) {}

  virtual void
  visitInfiniteLoopExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitPredicateLoopExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitPredicatePatternLoopExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitIteratorLoopExpression(std::shared_ptr<ast::Expression> let) {}

  virtual void visitLoopExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitIMatchExpression(std::shared_ptr<ast::Expression> let) {}

  virtual void visitBorrowExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitDereferenceExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitErrorPropagationExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitNegationExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void
  visitArithmeticOrLogicalExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitComparisonExpression(std::shared_ptr<ast::Expression> let) {
  }
  virtual void
  visitLazyBooleanExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitTypeCastExpression(std::shared_ptr<ast::Expression> let) {}
  virtual void visitAssignmentExpression(std::shared_ptr<ast::Expression> let) {
  }
  virtual void
  visitCompaoundAssignmentExpression(std::shared_ptr<ast::Expression> let) {}
};

void run(std::shared_ptr<ast::BlockExpression> block,
         BlockExpressionVisitor *visitor);

} // namespace rust_compiler::sema

// FIXME
