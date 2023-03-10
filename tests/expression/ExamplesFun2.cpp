#include "AST/Expression.h"
#include "AST/Item.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ExamplesFun2Test, CheckFun1) {

  std::string text =
      "pub fn add(right: usize) -> usize {  return if true { 5 } else { 6 } ;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun2) {

  std::string text =
      "fn add(right: usize) -> usize { return if true { 5 } else { 6 } }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun3) {

  std::string text = "fn add(right: usize) -> usize { return if true { 5 } }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun4) {

  std::string text = "return if true { 5 }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseReturnExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun5) {

  std::string text = "return if true { 5 } else { 6 }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseReturnExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun6) {

  std::string text = "fn add(right: usize) -> usize {  return 5; };";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun7) {

  std::string text = "fn add(right: usize) -> usize { };";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun8) {

  std::string text = "return if true { 5 } else {6}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseReturnExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun2Test, CheckFun9) {

  std::string text = "if true { 5 } else {6}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfExpression({});

  EXPECT_TRUE(result.isOk());
};
