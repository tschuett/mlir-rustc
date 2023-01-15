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

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> ifExpr =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(ifExpr.has_value());
};
