#include "Statement.h"

#include "BlockExpression.h"
#include "ExpressionStatement.h"
#include "ExpressionWithoutBlock.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "ReturnExpression.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(StatementTest, CheckStatement) {

  std::string text = "return left + right;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> stmt =
      parser.tryParseStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckReturnStatement) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> stmt =
      parser.tryParseStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckExpressionStatement) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> stmt =
      parser.tryParseExpressionStatement(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckExpressionWithoutBlock) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> stmt =
      parser.tryParseExpressionWithoutBlock(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};

TEST(StatementTest, CheckReturnExpression) {

  std::string text = "return 5";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> stmt =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(stmt.has_value());
};
