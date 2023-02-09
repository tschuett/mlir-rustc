#include "ArithmeticOrLogicalExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "Lexer/Lexer.h"
#include "Util.h"
#include <gtest/gtest.h>

#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ArithmeticOrLogicalExpressionTest, CheckOperatorExprSimple) {

  std::string text = "128 + 64";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> arith =
      parser.tryParseArithmeticOrLogicalExpresion(ts.getAsView());

  EXPECT_TRUE(arith.has_value());
};

TEST(ArithmeticOrLogicalExpressionTest, CheckOperatorExprSimple2) {

  std::string text = "foo + bar";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> arith =
      parser.tryParseArithmeticOrLogicalExpresion(ts.getAsView());

  EXPECT_TRUE(arith.has_value());
};

TEST(ArithmeticOrLogicalExpressionTest, CheckOperatorExprSimple3) {

  std::string text = "left + right";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> arith =
      parser.tryParseArithmeticOrLogicalExpresion(ts.getAsView());

  EXPECT_TRUE(arith.has_value());
};
