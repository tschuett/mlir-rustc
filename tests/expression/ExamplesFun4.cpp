#include "AST/Expression.h"
#include "AST/Item.h"
#include "AST/Statements.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ExamplesFun4Test, CheckFun0) {

  std::string text = "let mut i = 0;return 1;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> fun =
      parser.tryParseStatements(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun4Test, CheckFun01) {

  std::string text = "{let mut i = 0;return 1;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> fun =
      parser.tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun4Test, CheckFun1) {

  std::string text =
      "fn add(left: usize, right: usize) -> usize {let mut i = 0;return 1;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun4Test, CheckFun2) {

  std::string text =
      "fn add(left: usize, right: usize) -> usize {let mut i = 0;return 1;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun4Test, CheckFun3) {

  std::string text = "fn add(left: usize, right: usize) -> usize {let mut i = "
                     "0;while i < 10 {i = i + 1};return 1;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun4Test, CheckFun4) {

  std::string text = "fn add(left: usize, right: usize) -> usize {let mut i = "
                     "0;while i < 10 {i = i + 1};return 1;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};
