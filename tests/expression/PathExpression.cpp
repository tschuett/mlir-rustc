#include "Lexer/Lexer.h"
#include "PathInExpression.h"
#include "ReturnExpression.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(PathExpressionTest, CheckPathInExpr1) {

  std::string text = "super";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> pathIn =
      tryParsePathInExpression(ts.getAsView());

  EXPECT_TRUE(pathIn.has_value());
};

TEST(PathExpressionTest, CheckPathInExpr2) {

  std::string text = "foobar";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> pathIn =
      tryParsePathInExpression(ts.getAsView());

  EXPECT_TRUE(pathIn.has_value());
};

TEST(PathExpressionTest, CheckPathInExpr3) {

  std::string text = "::foo::bar";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> pathIn =
      tryParsePathInExpression(ts.getAsView());

  EXPECT_TRUE(pathIn.has_value());
};