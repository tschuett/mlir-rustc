#include "AST/Statement.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(LetStatementTest, CheckLetStatement1) {

  std::string text = "let i;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(LetStatementTest, CheckLetStatement122) {

  std::string text = "let i: i64;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(LetStatementTest, CheckLetStatement3) {

  std::string text = "let i: i64 = 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};

TEST(LetStatementTest, CheckLetStatement4) {

  std::string text = "let mut i: i64 = 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restriction;
  Result<std::shared_ptr<rust_compiler::ast::Statement>, std::string> result =
      parser.parseStatement(restriction);

  EXPECT_TRUE(result.isOk());
};
