#include "AST/Item.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ExamplesFun3Test, CheckFun1) {

  std::string text = "fn add(left: usize) -> usize {}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun3Test, CheckFun2) {

  std::string text = "fn add(left: usize) -> usize {let i;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun3Test, CheckFun3) {

  std::string text = "fn add(left: usize) -> usize {let i: i64}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun3Test, CheckFun4) {

  std::string text = "fn add(left: usize) -> usize {let i: i64 = 5}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun3Test, CheckFun5) {

  std::string text = "fn add(left: usize, right: usize) -> usize {let i: i64 = 5}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun3Test, CheckFun6) {

  std::string text = "fn add(left: usize, right: usize) -> usize {let i: i64 = 5;return i;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};
