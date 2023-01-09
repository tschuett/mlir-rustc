#include "Statement.h"

#include "ExpressionWithoutBlock.h"

#include "ReturnExpression.h"
#include "ExpressionStatement.h"
#include "BlockExpression.h"
#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(StatementTest, CheckStatement) {

  std::string text = "return left + right;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> stmt =
      tryParseStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckReturnStatement) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> stmt =
      tryParseStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckExpressionStatement) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> stmt =
      tryParseExpressionStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckExpressionWithoutBlock) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> stmt =
      tryParseExpressionWithoutBlock(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckReturnExpression) {

  std::string text = "return 5";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> stmt =
      tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};
