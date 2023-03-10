#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(FunctionTest, CheckFunctionReturnType1) {

  std::string text = "-> usize";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionReturnType, std::string> result =
      parser.parseFunctionReturnType();

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunctionReturnType2) {

  std::string text = "-> f32";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionReturnType, std::string> result =
      parser.parseFunctionReturnType();

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunctionQual1) {

  std::string text = "const async";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionQualifiers, std::string> result =
      parser.parseFunctionQualifiers();

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunctionParameters1) {

  std::string text = "right: usize, left: i128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionParameters, std::string> result =
      parser.parseFunctionParameters();

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunctionParam1) {

  std::string text = "right: usize";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionParameters, std::string> result =
      parser.parseFunctionParameters();

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunctionBody1) {

  std::string text = "{return right + right}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunctionBody2) {

  std::string text = "{return right + right;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFunction1) {

  std::string text = "fn add(right: usize) -> usize {return right + right;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};
