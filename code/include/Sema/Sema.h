#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/BlockExpression.h"
#include "AST/CallExpression.h"
#include "AST/Crate.h"
#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/LoopExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/Statements.h"
#include "AST/Types/Types.h"
#include "AST/MacroInvocationSemi.h"
#include "AST/LiteralExpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/MatchArmGuard.h"
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
  void walkOuterAttributes(std::shared_ptr<ast::OuterAttributes>);

  void analyzeFunction(std::shared_ptr<ast::Function> fun);
  void analyzeBlockExpression(std::shared_ptr<ast::BlockExpression> block);
  void analyzeStatements(std::shared_ptr<ast::Statements> stmts);
  void analyzeLetStatement(std::shared_ptr<ast::LetStatement> let);
  void analyzeCallExpression(std::shared_ptr<ast::CallExpression> let);
  void
  analyzeMethodCallExpression(std::shared_ptr<ast::MethodCallExpression> let);
  void
  analyzeExpressionStatement(std::shared_ptr<ast::ExpressionStatement> expr);
  void
  analyzeMacroInvocationSemi(std::shared_ptr<ast::MacroInvocationSemi> macro);
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

  //void checkExhaustiveness(std::shared_ptr<ast::MatchArmGuard>);

  bool isReachable(std::shared_ptr<ast::VisItem>,
                   std::shared_ptr<ast::VisItem>);

  basic::NodeId getNodeId(std::shared_ptr<ast::Node>);

  std::map<basic::ItemId, std::shared_ptr<ast::Item>> items;
  std::map<basic::NodeId, std::shared_ptr<ast::Node>> nodes;
  std::map<std::shared_ptr<ast::Node>, basic::NodeId> nodeIds;

  basic::NodeId nextId = 0;
};

void analyzeSemantics(std::shared_ptr<ast::Crate> &ast);

} // namespace rust_compiler::sema
