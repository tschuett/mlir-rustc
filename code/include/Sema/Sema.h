#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayExpression.h"
#include "AST/AssignmentExpression.h"
#include "AST/BlockExpression.h"
#include "AST/CallExpression.h"
#include "AST/CompoundAssignmentExpression.h"
#include "AST/ConstantItem.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/ItemDeclaration.h"
#include "AST/LetStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/LoopExpression.h"
#include "AST/MacroInvocationSemiStatement.h"
#include "AST/MatchArmGuard.h"
#include "AST/MatchExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "AST/Statement.h"
#include "AST/Statements.h"
#include "AST/StaticItem.h"
#include "AST/Struct.h"
#include "AST/StructExpression.h"
#include "AST/Trait.h"
#include "AST/Types/ImplTraitType.h"
#include "AST/Types/TraitObjectType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/Types.h"
#include "Basic/Ids.h"

#include <map>
#include <memory>

namespace rust_compiler::ast {
class StructStruct;
class TupleStruct;
} // namespace rust_compiler::ast

namespace rust_compiler::sema {

class Sema {

public:
  void analyze(std::shared_ptr<ast::Crate> &ast);

private:
  void walkItem(std::shared_ptr<ast::Item> item);
  void walkVisItem(std::shared_ptr<ast::VisItem> item);
  void walkOuterAttributes(std::span<ast::OuterAttribute>);

  void analyzeVisItem(std::shared_ptr<ast::VisItem> vis);
  // void analyzeFunction(std::shared_ptr<ast::Function> fun);
  void analyzeBlockExpression(ast::BlockExpression *block);
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
  void analyzeExpression(ast::Expression *let);
  void analyzeExpressionWithBlock(ast::ExpressionWithBlock *let);
  void analyzeExpressionWithoutBlock(ast::ExpressionWithoutBlock *let);

  void analyzeItemDeclaration(std::shared_ptr<ast::ItemDeclaration> item);

  void analyzeArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith);

  void analyzeLoopExpression(ast::LoopExpression *arith);
  void analyzeOperatorExpression(ast::OperatorExpression *arith);

  void analyzeLiteralExpression(std::shared_ptr<ast::LiteralExpression> arith);

  void analyzeInfiniteLoopExpression(ast::InfiniteLoopExpression *arith);
  void
  analyzePredicatePatternLoopExpression(ast::PredicatePatternLoopExpression *);

  void analyzeAssignmentExpression(ast::AssignmentExpression *arith);
  void analyzeCompoundAssignmentExpression(ast::CompoundAssignmentExpression *);
  void analyzeMatchExpression(ast::MatchExpression *);
  void analyzeIfLetExpression(ast::IfLetExpression *);
  void analyzeIfExpression(ast::IfExpression *);

  // void checkExhaustiveness(std::shared_ptr<ast::MatchArmGuard>);

  void analyzeArrayExpression(ast::ArrayExpression *);
  void analyzeConstantItem(ast::ConstantItem *);
  void analyzeStaticItem(ast::StaticItem *);
  void analyzeTrait(ast::Trait *);
  void analyzeFunction(ast::Function *);
  void analyzeStruct(ast::Struct *);
  void analyzeStructStruct(ast::StructStruct *);
  void analyzeTupleStruct(ast::TupleStruct *);

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

  // types
  void walkType(ast::types::TypeExpression *);
  void walkTypeNoBounds(ast::types::TypeNoBounds *);

  std::pair<size_t, size_t> getAlignmentAndSizeOfType(ast::types::TypeExpression*);
  std::pair<size_t, size_t> getAlignmentAndSizeOfTypeNoBounds(ast::types::TypeNoBounds*);
  std::pair<size_t, size_t> getAlignmentAndSizeOfImplTraitType(ast::types::ImplTraitType*);
  std::pair<size_t, size_t> getAlignmentAndSizeOfTraitObjectType(ast::types::TraitObjectType*);

  bool isReprAttribute(const ast::SimplePath&) const;
};

void analyzeSemantics(std::shared_ptr<ast::Crate> &ast);

} // namespace rust_compiler::sema
