#pragma once

namespace rust_compiler::ast {
class Expression;
class ExpressionWithoutBlock;
class Statement;
class LoopExpression;
class PredicatePatternLoopExpression;
class OperatorExpression;
class BlockExpression;
class ExpressionStatement;
class ExpressionWithBlock;
class IteratorLoopExpression;
class ClosureExpression;
class ArrayExpression;
} // namespace rust_compiler::ast

namespace rust_compiler::sema {

class ReturnExpressionSearcher {
  bool foundReturn = false;

public:
  bool containsReturnExpression(ast::BlockExpression *block);

private:
  void visitStatement(ast::Statement *stmt);
  void visitExpressionStatement(ast::ExpressionStatement *stmt);
  void visitExpressionWithBlock(ast::ExpressionWithBlock *stmt);
  void visitExpressionWithoutBlock(ast::ExpressionWithoutBlock *stmt);
  void visitBlockExpression(ast::BlockExpression *stmt);
  void visitExpression(ast::Expression *stmt);
  void visitOperatorExpression(ast::OperatorExpression *stmt);
  void visitLoopExpression(ast::LoopExpression *stmt);
  void visitPredicatePatternLoopExpression(
      ast::PredicatePatternLoopExpression *stmt);
  void visitIteratorLoopExpression(ast::IteratorLoopExpression *);
  void visitClosureExpression(ast::ClosureExpression *);
  void visitArrayExpression(ast::ArrayExpression *);
};

bool containsReturnExpression(ast::BlockExpression *block);

} // namespace rust_compiler::sema
