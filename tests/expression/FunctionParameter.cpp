#include "gtest/gtest.h"
#include "FunctionParameter.h"
#include "FunctionParameters.h"
#include "Lexer/Lexer.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;


TEST(ExpressionTest, CheckFunctionParameter) {

  std::string text = "left: u128";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionParameter> param =
      tryParseFunctionParameter(ts.getAsView());

  EXPECT_TRUE(param.has_value());
};

TEST(ExpressionTest, CheckFunctionParameters) {

  std::string text = "left: u128, right: i8";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::FunctionParameters> params =
      tryParseFunctionParameters(ts.getAsView());

  EXPECT_TRUE(params.has_value());
};
