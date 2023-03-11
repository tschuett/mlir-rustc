#include "AST/FunctionParam.h"
#include "AST/FunctionParameters.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(FunctionParameterExpressionTest, CheckFunctionParameter) {

  std::string text = "left: u128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionParam, std::string> result =
      parser.parseFunctionParam();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionParameterExpressionTest, CheckFunctionParameters) {

  std::string text = "left: u128, right: i8";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionParameters, std::string> result =
      parser.parseFunctionParameters();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};
