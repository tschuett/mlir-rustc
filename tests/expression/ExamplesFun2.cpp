#include "AST/Expression.h"
#include "AST/Item.h"
#include "BlockExpression.h"
#include "ExpressionStatement.h"
#include "Function.h"
#include "Item.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "ReturnExpression.h"
#include "Statement.h"
#include "gtest/gtest.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ExamplesFun2Test, CheckFun1) {

  std::string text =
      "pub fn add(right: usize) -> usize {  return if true { 5 } else { 6 } }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Item>> fun =
      parser.tryParseItem(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun2) {

  std::string text =
      "fn add(right: usize) -> usize { return if true { 5 } else { 6 } }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun2a) {

  std::string text =
      "fn add(right: usize) -> usize { return if true { 5 } }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun2ab) {

  std::string text =
      "return if true { 5 }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> fun =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun2ac) {

  std::string text =
      "return if true { 5 } else { 6 }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> fun =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun2b) {

  std::string text =
      "fn add(right: usize) -> usize {  return 5; };";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun3) {

  std::string text =
      "fn add(right: usize) -> usize { };";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<rust_compiler::ast::Function> fun =
      parser.tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun4) {

  std::string text = "return if true { 5 } else {6}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> fun =
      parser.tryParseReturnExpression(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};

TEST(ExamplesFun2Test, CheckFun5) {

  std::string text = "if true { 5 } else {6}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> fun =
      parser.tryParseIfExpression(ts.getAsView());

  EXPECT_TRUE(fun.has_value());
};
