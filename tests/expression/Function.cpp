#include "Function.h"

#include "BlockExpression.h"
#include "FunctionParam.h"
#include "FunctionParameters.h"
#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(FunctionTest, CheckFunctionQual1) {

  std::string text = "const async";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionQualifiers> fun =
      tryParseFunctionQualifiers(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 2;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionParameters1) {

  std::string text = "right: usize, left: i128";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionParameters> fun =
      tryParseFunctionParameters(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 7;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionParam1) {

  std::string text = "right: usize";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionParam> fun =
      tryParseFunctionParam(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 3;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionSig0) {

  std::string text = "fn add()";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionSignature> fun =
      tryParseFunctionSignature(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 4;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionSig0b) {

  std::string text = "fn add() -> usize";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionSignature> fun =
      tryParseFunctionSignature(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 6;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionSig1a) {

  std::string text = "fn add(right: usize)";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionSignature> fun =
      tryParseFunctionSignature(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 7;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionSig1b) {

  std::string text = "fn add(right: usize) -> usize";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionSignature> fun =
      tryParseFunctionSignature(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 9;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionSig2) {

  std::string text = "fn add(right: usize, left: i128)";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionSignature> fun =
      tryParseFunctionSignature(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 11;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionSig3) {

  std::string text = "fn add(right: usize, left: i128) -> usize";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionSignature> fun =
      tryParseFunctionSignature(ts.getAsView());

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 13;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionBody1) {

  std::string text = "{return right + right}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());

  size_t expectedLendth = 6;

  EXPECT_EQ((*block)->getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunctionBody2) {

  std::string text = "{return right + right;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());

  size_t expectedLendth = 7;

  EXPECT_EQ((*block)->getTokens(), expectedLendth);
};

TEST(FunctionTest, CheckFunction1) {

  std::string text = "fn add(right: usize) -> usize {return right + right;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Function> fun =
      tryParseFunction(ts.getAsView(), "");

  EXPECT_TRUE(fun.has_value());

  size_t expectedLendth = 16;

  EXPECT_EQ((*fun).getTokens(), expectedLendth);
};