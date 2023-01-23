#include "AST/Statement.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(LetStatementTest, CheckLetStatement1) {

  std::string text = "let i;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> loop =
      parser.tryParseLetStatement(ts.getAsView());

  EXPECT_TRUE(loop.has_value());
};

TEST(LetStatementTest, CheckLetStatement2) {

  std::string text = "let i: i64;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> loop =
      parser.tryParseLetStatement(ts.getAsView());

  EXPECT_TRUE(loop.has_value());
};

TEST(LetStatementTest, CheckLetStatement3) {

  std::string text = "let i: i64 = 5;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> loop =
      parser.tryParseLetStatement(ts.getAsView());

  EXPECT_TRUE(loop.has_value());
};
