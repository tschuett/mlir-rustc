#include "Lexer/Lexer.h"
#include "ReturnExpression.h"
#include "LiteralExpression.h"
#include "gtest/gtest.h"

#include "AST/ReturnExpression.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ExpressionTest, CheckReturnExpr) {

  std::string text = "return left + right";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ret =
      tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(ret.has_value());
};



TEST(ExpressionTest, CheckLiteralExpr) {

  std::string text = "128";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> lit =
      tryParseLiteralExpression(ts.getAsView());

  EXPECT_TRUE(lit.has_value());
};
