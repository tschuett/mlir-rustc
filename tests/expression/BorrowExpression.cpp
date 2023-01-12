#include "BlockExpression.h"
#include "ExpressionStatement.h"
#include "ExpressionWithoutBlock.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "ReturnExpression.h"
#include "Statement.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(BorrowExpressionTest, CheckBorrowExpr1) {

  std::string text = "&foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> borrow =
    parser.tryParseBorrowExpression(ts.getAsView());
 
  EXPECT_TRUE(borrow.has_value());
};
