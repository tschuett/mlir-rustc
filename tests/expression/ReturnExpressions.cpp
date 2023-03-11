#include "AST/FunctionParam.h"
#include "AST/ReturnExpression.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ReturnExpressionTest, CheckReturnExpr) {

  std::string text = "return left + right";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 5;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseReturnExpression({});

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(ReturnExpressionTest, CheckReturnExpr1) {

  std::string text = "return left";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseReturnExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ReturnExpressionTest, CheckReturnExpr2) {

  std::string text = "return";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseReturnExpression({});

  EXPECT_TRUE(result.isOk());
};
