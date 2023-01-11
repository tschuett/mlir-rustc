#include "AST/OperatorExpression.h"

#include "Lexer/Lexer.h"
#include "OperatorExpression.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(OperatorExpressionTest, CheckOperatorExprSimple) {

  std::string text = "128 + 64";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> op =
      parser.tryParseOperatorExpression(ts.getAsView());

  EXPECT_TRUE(op.has_value());
};

TEST(OperatorExpressionTest, CheckOperatorExprSimple2) {

  std::string text = "foo + bar";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> op =
      parser.tryParseOperatorExpression(ts.getAsView());

  EXPECT_TRUE(op.has_value());
};

TEST(OperatorExpressionTest, CheckOperatorExprSimple3) {

  std::string text = "left + right";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> op =
      parser.tryParseOperatorExpression(ts.getAsView());

  EXPECT_TRUE(op.has_value());
};
