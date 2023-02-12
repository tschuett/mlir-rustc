#include "AST/Statements.h"

#include "Parser/Parser.h"

#include "Lexer/Lexer.h"
#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

//TEST(StatementsTest, CheckStatements1) {
//
//  std::string text = "{return left + right;}";
//
//  TokenStream ts = lex(text, "lib.rs");
//
//  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> stmts =
//      tryParseStatements(ts.getAsView());
//
//  EXPECT_TRUE(stmts.has_value());
//};
//
//TEST(StatementsTest, CheckStatements2) {
//
//  std::string text = "return left + right;";
//
//  TokenStream ts = lex(text, "lib.rs");
//
//  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> stmts =
//      tryParseStatements(ts.getAsView());
//
//  EXPECT_TRUE(stmts.has_value());
//};

TEST(StatementsTest, CheckFun0a) {

  std::string text = "let mut i = 0;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> fun =
      parser.tryParseStatement(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(StatementsTest, CheckFun0c) {

  std::string text = "let mut i = 0;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> fun =
      parser.tryParseStatements(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(StatementsTest, CheckFun0d) {

  std::string text = "let mut i = 0;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> fun =
      parser.tryParseLetStatement(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(StatementsTest, CheckFun1) {

  std::string text = "let mut i = 0;return 1";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> fun =
      parser.tryParseStatements(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(StatementsTest, CheckFun2) {

  std::string text = "let mut i = 0;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> fun =
      parser.tryParseStatement(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(StatementsTest, CheckFun3) {

  std::string text = "let mut i = 0;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Statement>> fun =
      parser.tryParseLetStatement(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};
