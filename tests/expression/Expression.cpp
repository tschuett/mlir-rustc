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

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseLiteralExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionTest, CheckNegationExpr1) {

  std::string text = "!128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseNegationExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionTest, CheckNegationExpr2) {

  std::string text = "-128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseNegationExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionTest, CheckExprSmaller) {

  std::string text = "i < 10";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  rust_compiler::parser::Restrictions restrictions;
  StringResult<std::shared_ptr<rust_compiler::ast::Expression>> result =
      parser.parseExpression({}, restrictions);


  EXPECT_TRUE(result.isOk());
};
