#include "AST/FunctionParameter.h"
#include "AST/ReturnExpression.h"
#include "BlockExpression.h"
#include "Lexer/Lexer.h"
#include "LiteralExpression.h"
#include "Statement.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;


TEST(ExpressionTest, CheckLiteralExpr) {

  std::string text = "128";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> lit =
      tryParseLiteralExpression(ts.getAsView());

  EXPECT_TRUE(lit.has_value());
};


TEST(ExpressionTest, CheckBlockExpr) {

  std::string text = "{return left + right;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};

TEST(ExpressionTest, CheckStatement) {

  std::string text = "return left + right;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> stmt =
      tryParseStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

