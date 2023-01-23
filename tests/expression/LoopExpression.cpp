#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(LoopExpressionTest, CheckLoopExpr1) {

  std::string text = "while i < 10 {i = i + 1}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> loop =
      parser.tryParsePredicateLoopExpression(ts.getAsView());

  size_t expected = 11;

  EXPECT_TRUE(loop.has_value());

  EXPECT_EQ(expected, (*loop)->getTokens());
};

TEST(LoopExpressionTest, CheckLoopExpr2) {

  std::string text = "let mut i = 0; while i < 10 {i = i + 1}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> loop =
      parser.tryParsePredicateLoopExpression(ts.getAsView());

  size_t expected = 17;

  EXPECT_TRUE(loop.has_value());

  EXPECT_EQ(expected, (*loop)->getTokens());
};


