#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ClippyAttribute.h"
#include "AST/InnerAttribute.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "AST/UseDeclaration.h"
#include "AST/UseTree.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"

#include <span>
#include <string>
#include <string_view>

namespace rust_compiler::parser {

class Parser {
  lexer::TokenStream ts;
  std::string modulePath;

public:
  Parser(lexer::TokenStream &ts, std::string_view modulePath)
      : ts(ts), modulePath(modulePath){};

  std::shared_ptr<ast::Module> parse();

  // private:

  std::optional<std::shared_ptr<ast::Item>>
  tryParseItem(std::span<lexer::Token> tokens, std::string_view modulePath);

  std::optional<ast::Module> tryParseModule(std::span<lexer::Token> tokens,
                                            std::string_view modulePath);

  std::optional<ast::Module> tryParseModuleTree(std::span<lexer::Token> tokens,
                                                std::string_view modulePath);

  std::optional<ast::Visibility>
  tryParseVisibility(std::span<lexer::Token> tokens);

  std::optional<ast::SimplePath>
  tryParseSimplePath(std::span<lexer::Token> tokens);

  std::optional<ast::Function> tryParseFunction(std::span<lexer::Token> tokens,
                                                std::string_view modulePath);

  std::optional<ast::FunctionQualifiers>
  tryParseFunctionQualifiers(std::span<lexer::Token> tokens);

  std::optional<ast::FunctionSignature>
  tryParseFunctionSignature(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParseFunctionReturnType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::BlockExpression>>
  tryParseBlockExpression(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Statement>>
      tryParseStatement(std::span<lexer::Token>);

  std::optional<std::shared_ptr<ast::Expression>>
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

  std::optional<ast::OuterAttribute>
  tryParseOuterAttribute(std::span<lexer::Token> tokens);

  std::optional<ast::InnerAttribute>
  tryParseInnerAttribute(std::span<lexer::Token> tokens);

  std::optional<ast::ClippyAttribute>
  tryParseClippyAttribute(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::types::Type>>
  tryParsePrimitiveType(std::span<lexer::Token> tokens);

  std::optional<std::shared_ptr<ast::Expression>>
tryParseNegationExpression(std::span<lexer::Token> tokens);

};

} // namespace rust_compiler::parser

// FIXME scope path resp. stack?
