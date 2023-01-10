#include "BlockExpression.h"
#include "ExpressionStatement.h"
#include "Item.h"
#include "AST/Item.h"
#include "Lexer/Lexer.h"
#include "ReturnExpression.h"
#include "Statement.h"
#include "gtest/gtest.h"
#include "Function.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ExamplesFun1Test, CheckFun5) {

  std::string text =
      "fn add() { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
    tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun1Test, CheckFun4a) {

  std::string text =
      "fn add(left: usize) { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
    tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun1Test, CheckFun4) {

  std::string text =
      "fn add(left: usize, right: usize) { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
    tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun1Test, CheckFun3a) {

  std::string text =
      "fn add() -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
    tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun1Test, CheckFun3b) {

  std::string text =
      "fn add(right: usize) -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
    tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun1Test, CheckFun3c) {

  std::string text =
      "fn add(left: usize, right: usize) -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
    tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun1Test, CheckFun1) {

  std::string text =
      "pub fn add(left: usize, right: usize) -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Item>> item =
      tryParseItem(ts.getAsView(), "");

  EXPECT_TRUE(item.has_value());
};

TEST(ExamplesFun1Test, CheckFun2) {

  std::string text =
      "pub fn add(left: usize, right: usize) -> usize { return left + right;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Item>> item =
      tryParseItem(ts.getAsView(), "");

  EXPECT_TRUE(item.has_value());
};


