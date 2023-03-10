#include "AST/Statement.h"

#include "ADT/CanonicalPath.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(StatementTest, CheckReturnStatement) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckStatement) {

  std::string text = "return left + right;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckExpressionStatement) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckExpressionWithoutBlock) {

  std::string text = "return 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckReturnExpression) {

  std::string text = "return 5";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parseReturnExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckLetStatement) {

  std::string text = "let i: i64 = 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckLetStatement1) {

  std::string text = "let i: i64;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(StatementTest, CheckLetStatement2) {

  std::string text = "let i;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};
