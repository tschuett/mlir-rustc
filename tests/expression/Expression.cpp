#include "AST/ReturnExpression.h"
#include "Lexer/Lexer.h"
#include "LiteralExpression.h"
#include "PathInExpression.h"
#include "ReturnExpression.h"
#include "gtest/gtest.h"

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

TEST(ExpressionTest, CheckPathInExpr1) {

  std::string text = "super";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> pathIn =
      tryParsePathInExpression(ts.getAsView());

  EXPECT_TRUE(pathIn.has_value());
};

TEST(ExpressionTest, CheckPathInExpr2) {

  std::string text = "foobar";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> pathIn =
      tryParsePathInExpression(ts.getAsView());

  EXPECT_TRUE(pathIn.has_value());
};

TEST(ExpressionTest, CheckPathInExpr3) {

  std::string text = "::foo::bar";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> pathIn =
      tryParsePathInExpression(ts.getAsView());

  EXPECT_TRUE(pathIn.has_value());
};
