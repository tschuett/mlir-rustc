#include "FunctionParam.h"
#include "FunctionParameters.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(FunctionParameterExpressionTest, CheckFunctionParameter) {

  std::string text = "left: u128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<rust_compiler::ast::FunctionParam> param =
      parser.tryParseFunctionParam(ts.getAsView());

  EXPECT_TRUE(param.has_value());
};

TEST(FunctionParameterExpressionTest, CheckFunctionParameters) {

  std::string text = "left: u128, right: i8";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<rust_compiler::ast::FunctionParameters> params =
      parser.tryParseFunctionParameters(ts.getAsView());

  EXPECT_TRUE(params.has_value());
};
