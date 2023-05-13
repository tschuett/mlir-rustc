#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/BlockExpression.h"
#include "AST/CallExpression.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/IfLetExpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/LetStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/LoopExpression.h"
#include "AST/MacroInvocationSemiStatement.h"
#include "AST/MatchArmGuard.h"
#include "AST/MethodCallExpression.h"
#include "AST/Statement.h"
#include "AST/Statements.h"
#include "AST/StructExpression.h"
#include "AST/Types/Types.h"
#include "Basic/Ids.h"

#include <map>
#include <memory>

namespace rust_compiler::sema {

class Sema {

public:
  void analyze(std::shared_ptr<ast::Crate> &ast);

private:
  void walkItem(std::shared_ptr<ast::Item> item);
  void walkVisItem(std::shared_ptr<ast::VisItem> item);
  void walkOuterAttributes(std::span<ast::OuterAttribute>);

  void analyzeVisItem(std::shared_ptr<ast::VisItem> vis);
  void analyzeFunction(std::shared_ptr<ast::Function> fun);
  void analyzeBlockExpression(std::shared_ptr<ast::BlockExpression> block);
  void analyzeStatements(ast::Statements stmts);
  void analyzeLetStatement(std::shared_ptr<ast::LetStatement> let);
  void analyzeCallExpression(std::shared_ptr<ast::CallExpression> let);
  void
  analyzeMethodCallExpression(std::shared_ptr<ast::MethodCallExpression> let);
  void
  analyzeExpressionStatement(std::shared_ptr<ast::ExpressionStatement> expr);
  void analyzeMacroInvocationSemiStatement(
      std::shared_ptr<ast::MacroInvocationSemiStatement> macro);
  //  void analyzeMacroInvocationSemiStatement(
  //      std::shared_ptr<ast::MacroInvocationSemiStatement> macro);
  void analyzeExpression(std::shared_ptr<ast::Expression> let);
  void
  analyzeExpressionWithBlock(std::shared_ptr<ast::ExpressionWithBlock> let);
  void analyzeExpressionWithoutBlock(
      std::shared_ptr<ast::ExpressionWithoutBlock> let);

  void analyzeItemDeclaration(std::shared_ptr<ast::Node> item);

  void analyzeArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith);

  void analyzeLoopExpression(std::shared_ptr<ast::LoopExpression> arith);
  void
  analyzeOperatorExpression(std::shared_ptr<ast::OperatorExpression> arith);

  void analyzeLiteralExpression(std::shared_ptr<ast::LiteralExpression> arith);

  void analyzeInfiniteLoopExpression(
      std::shared_ptr<ast::InfiniteLoopExpression> arith);

  void
  analyzeAssignmentExpression(std::shared_ptr<ast::AssignmentExpression> arith);

  // void checkExhaustiveness(std::shared_ptr<ast::MatchArmGuard>);

  void analyzeArrayExpression(std::shared_ptr<ast::ArrayExpression>);

  bool isReachable(std::shared_ptr<ast::VisItem>,
                   std::shared_ptr<ast::VisItem>);

  /// This should probably run after type checking.
  /// https://doc.rust-lang.org/reference/const_eval.html
  bool isConstantExpression(ast::Expression *);
  bool isConstantEpressionWithoutBlock(ast::ExpressionWithoutBlock *);
  bool isConstantEpressionWithBlock(ast::ExpressionWithBlock *);
  bool isConstantOperatorExpression(ast::OperatorExpression *);
  bool isConstantBlockExpression(ast::BlockExpression *op);
  bool isConstantStatement(ast::Statement *stmt);
  bool isConstantLoopExpression(ast::LoopExpression *);
  bool isConstantStructExpression(ast::StructExpression *);
  bool isConstantIfLetExpression(ast::IfLetExpression *);

  /// https://doc.rust-lang.org/reference/expressions.html#place-expressions-and-value-expressions
  bool isPlaceExpression(ast::Expression *);
  bool isPlaceExpressionWithBlock(ast::ExpressionWithBlock *);
  bool isPlaceExpressionWithoutBlock(ast::ExpressionWithoutBlock *);
  bool isPlaceOperatorExpression(ast::OperatorExpression *);

  bool isAssigneeExpression(ast::Expression *);
  bool isAssigneeExpressionWithBlock(ast::ExpressionWithBlock *);
  bool isAssigneeExpressionWithoutBlock(ast::ExpressionWithoutBlock *);

  bool isValueExpression(ast::Expression *);
  bool isValueExpressionWithBlock(ast::ExpressionWithBlock *);
  bool isValueExpressionWithoutBlock(ast::ExpressionWithoutBlock *);
};

void analyzeSemantics(std::shared_ptr<ast::Crate> &ast);

} // namespace rust_compiler::sema
