#include "Lexer/Lexer.h"
#include "ReturnExpression.h"
#include "gtest/gtest.h"

#include "AST/ReturnExpression.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(TypesTest, CheckReturnExpr) {

  std::string text = "return left + right";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::ReturnExpression> ret =
      tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(ret.has_value());
};
