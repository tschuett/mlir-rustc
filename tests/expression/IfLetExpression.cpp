#include "AST/IfLetExpression.h"

#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"
#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(IfLetExpressionTest, CheckIfLetExpression) {

  std::string text = "if let (a) = 4 { 4; }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 12;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfLetExpression({});

  EXPECT_TRUE(result.isOk());
};
