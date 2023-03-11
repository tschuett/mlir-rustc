#include "AST/OperatorExpression.h"

#include "ADT/CanonicalPath.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(OperatorExpressionTest, CheckOperatorExprSimple9) {

  std::string text = "128 + 64";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 4;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  rust_compiler::parser::Restrictions restrictions;
  StringResult<std::shared_ptr<rust_compiler::ast::Expression>> result =
      parser.parseExpression({}, restrictions);

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(OperatorExpressionTest, CheckOperatorExprSimple2) {

  std::string text = "foo + bar";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 4;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  rust_compiler::parser::Restrictions restrictions;
  StringResult<std::shared_ptr<rust_compiler::ast::Expression>> result =
      parser.parseExpression({}, restrictions);

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(OperatorExpressionTest, CheckOperatorExprSimple3) {

  std::string text = "left + right";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 4;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  rust_compiler::parser::Restrictions restrictions;
  StringResult<std::shared_ptr<rust_compiler::ast::Expression>> result =
      parser.parseExpression({}, restrictions);

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};
