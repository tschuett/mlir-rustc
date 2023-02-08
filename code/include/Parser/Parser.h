#pragma once

#include "ADT/ScopedCanonicalPath.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/GenericArgs.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/InnerAttribute.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "AST/OuterAttributes.h"
#include "AST/PathExprSegment.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/Patterns.h"
#include "AST/Patterns/RestPattern.h"
#include "AST/Patterns/TuplePattern.h"
#include "AST/Scrutinee.h"
#include "AST/Types/TypeExpression.h"
#include "AST/UseDeclaration.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"

#include <mlir/Support/LogicalResult.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast::use_tree {
class PathList;
class UseTree;
} // namespace rust_compiler::ast::use_tree

namespace rust_compiler::ast {
class Statements;
class Statement;
class Crate;
} // namespace rust_compiler::ast

namespace rust_compiler::parser {

class Parser {
  lexer::TokenStream ts;
  adt::ScopedCanonicalPath path;

public:
  Parser(lexer::TokenStream &ts, const adt::CanonicalPath &path)
      : ts(ts), path(path){};

  mlir::LogicalResult parseFile(std::shared_ptr<ast::Module> &);

  // private:

  std::optional<std::shared_ptr<ast::Item>>
  tryParseItem(std::span<lexer::Token> tokens);

  std::optional<ast::Module> tryParseModule(std::span<lexer::Token> tokens);

  std::optional<ast::Module> tryParseModuleTree(std::span<lexer::Token> tokens,
                                                std::string_view moduleName);

  std::optional<ast::Visibility>
  tryParseVisibility(std::span<lexer::Token> tokens);

  std::optional<ast::SimplePath>
  tryParseSimplePath(std::span<lexer::Token> tokens);

  std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens);

  std::optional<ast::FunctionQualifiers>
  tryParseFunctionQualifiers(std::span<lexer::Token> tokens);

  std::optional<ast::FunctionSignature>
  tryParseFunctionSignature(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::TypeExpression>>
  tryParseFunctionReturnType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::BlockExpression>>
  tryParseBlockExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Statement>>
      tryParseStatement(std::span<lexer::Token>);

  std::optional<std::shared_ptr<ast::Statement>>
  tryParseExpressionStatement(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseExpressionWithBlock(std::span<lexer::Token> view);

  std::optional<std::shared_ptr<ast::Statements>>
  tryParseStatements(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseExpressionWithoutBlock(std::span<lexer::Token> view);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseLiteralExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseOperatorExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseOperatorFeedingExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseArithmeticOrLogicalExpresion(std::span<lexer::Token> tokens);

  std::optional<ast::ArithmeticOrLogicalExpressionKind>
  tryParserOperator(std::span<lexer::Token> tokens);

  std::optional<ast::FunctionParameters>
  tryParseFunctionParameters(std::span<lexer::Token> tokens);

  std::optional<ast::FunctionParam>
  tryParseFunctionParam(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
      tryParseExpression(std::span<lexer::Token>);

  std::optional<ast::UseDeclaration>
  tryParseUseDeclaration(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<rust_compiler::ast::use_tree::UseTree>>
  tryParseUseTree(std::span<lexer::Token> tokens);

  std::optional<ast::use_tree::PathList>
  tryParsePathList(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseReturnExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParseType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::OuterAttribute>>
  tryParseOuterAttribute(std::span<lexer::Token> tokens);

  std::optional<ast::InnerAttribute>
  tryParseInnerAttribute(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParsePrimitiveType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseNegationExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseBorrowExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::OuterAttributes>>
  tryParseOuterAttributes(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::VisItem>>
  tryParseVisItem(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseIfExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseIfLetExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::Pattern>>
  tryParsePattern(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Scrutinee>>
  tryParseScrutinee(std::span<lexer::Token> tokens);

  std::optional<ast::WhereClause>
  tryParseWhereClause(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParsePathExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParsePathInExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParseQualifiedPathInExpression(std::span<lexer::Token> tokens);

  std::optional<ast::PathExprSegment>
  tryParsePathExprSegment(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::PatternWithoutRange>>
  tryParseIdentifierPattern(std::span<lexer::Token> tokens);

  std::optional<ast::PathIdentSegment>
  tryParsePathIdentSegment(std::span<lexer::Token> tokens);

  std::optional<ast::GenericArgs>
  tryParseGenericArgs(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  tryParsePatternNoTopAlt(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  tryParseLiteralPattern(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::SelfParam>>
  tryParseShorthandSelf(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::SelfParam>>
  tryParseTypedSelf(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  tryParseTuplePattern(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::TuplePatternItems>>
  tryParseTuplePatternItems(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
  tryParseRestPattern(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParseQualifiedPathType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParseQualifiedPathInType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParseTypePath(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
  tryParsePredicateLoopExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Statement>>
      tryParseLetStatement(std::span<lexer::Token>);

  std::optional<std::shared_ptr<ast::Expression>>
      tryParseComparisonExpression(std::span<lexer::Token>);

  std::optional<std::shared_ptr<ast::Expression>>
      tryParseAssignmentExpression(std::span<lexer::Token>);

  std::optional<std::shared_ptr<ast::Expression>>
      tryParseAwaitExpression(std::span<lexer::Token>);

  std::optional<std::shared_ptr<ast::types::TypeExpression>>
      tryParseTypeExpression(std::span<lexer::Token>);

  void printToken(lexer::Token &);
};

} // namespace rust_compiler::parser

// FIXME scope path resp. stack?
