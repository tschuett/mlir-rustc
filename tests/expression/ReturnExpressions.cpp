#include "AST/FunctionParam.h"
#include "AST/ReturnExpression.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ReturnExpressionTest, CheckReturnExpr) {

  std::string text = "return left + right";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 4;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ret =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(ret.has_value());
};

TEST(ReturnExpressionTest, CheckReturnExpr1) {

  std::string text = "return left";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ret =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(ret.has_value());
};

TEST(ReturnExpressionTest, CheckReturnExpr2) {

  std::string text = "return";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ret =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(ret.has_value());
};
