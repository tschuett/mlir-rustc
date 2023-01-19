#include "AST/IfExpression.h"

#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(IfExpressionTest, CheckIfExpression) {

  std::string text = "if 5 { 4 }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 5;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ifExpr =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(ifExpr.has_value());
};

TEST(IfExpressionTest, CheckIfExpression1) {

  std::string text = "if 5 { 4 } else { 5 }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 9;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ifExpr =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(ifExpr.has_value());
};

TEST(IfExpressionTest, CheckIfExpression2) {

  std::string text = "if 5 { 4 } else if true { 5 }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 11;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ifExpr =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(ifExpr.has_value());
};

TEST(IfExpressionTest, CheckIfExpression3) {

  std::string text = "if 5 { 4 } else if true { 5 } else { 6 }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 15;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ifExpr =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(ifExpr.has_value());
};

TEST(IfExpressionTest, CheckIfExpression4) {

  std::string text = "if 5 { 4 } else if true { 5 } if true { 6 } else { 6 }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 20;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ifExpr =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(ifExpr.has_value());
};
