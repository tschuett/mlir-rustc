#include "ArithmeticOrLogicalExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "Lexer/Lexer.h"
#include "Util.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ArithmeticOrLogicalExpressionTest, CheckOperatorExprSimple) {

  std::string text = "128 + 64";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> arith =
      tryParseArithmeticOrLogicalExpresion(ts.getAsView());

  EXPECT_TRUE(arith.has_value());
};

TEST(ArithmeticOrLogicalExpressionTest, CheckOperatorExprSimple2) {

  std::string text = "foo + bar";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> arith =
      tryParseArithmeticOrLogicalExpresion(ts.getAsView());

  EXPECT_TRUE(arith.has_value());
};

TEST(ArithmeticOrLogicalExpressionTest, CheckOperatorExprSimple3) {

  std::string text = "left + right";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> arith =
      tryParseArithmeticOrLogicalExpresion(ts.getAsView());

  EXPECT_TRUE(arith.has_value());
};
