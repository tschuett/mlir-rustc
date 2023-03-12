#include "AST/Item.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ExamplesFun1Test, CheckFun5) {

  std::string text = "fn add() { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun4a) {

  std::string text = "fn add(left: usize) { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun4) {

  std::string text = "fn add(left: usize, right: usize) { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun3a) {

  std::string text = "fn add() -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun3b) {

  std::string text = "fn add(right: usize) -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun3c) {

  std::string text = "fn add(left: usize, right: usize) -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun1) {

  std::string text = "fn add(left: usize, right: usize) -> usize { }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckFun2) {

  std::string text =
      "fn add(right: usize) -> usize { return right + right;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(ExamplesFun1Test, CheckModule1) {

  std::string text =
      "pub fn add(right: usize) -> usize { return right + right;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Crate>, std::string> result =
      parser.parseCrateModule("crateName", 5);

  EXPECT_TRUE(result.isOk());
};
