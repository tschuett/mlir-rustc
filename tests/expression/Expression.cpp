#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ExpressionTest, CheckLiteralExpr) {

  std::string text = "128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> lit =
      parser.tryParseLiteralExpression(ts.getAsView());

  EXPECT_TRUE(lit.has_value());
};

TEST(ExpressionTest, CheckNegationExpr1) {

  std::string text = "!128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> neg =
      parser.tryParseNegationExpression(ts.getAsView());

  EXPECT_TRUE(neg.has_value());
};

TEST(ExpressionTest, CheckNegationExpr2) {

  std::string text = "-128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> neg =
      parser.tryParseNegationExpression(ts.getAsView());

  EXPECT_TRUE(neg.has_value());
};

TEST(ExpressionTest, CheckExprSmaller) {

  std::string text = "i < 10";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> neg =
      parser.tryParseExpression(ts.getAsView());

  EXPECT_TRUE(neg.has_value());
};
