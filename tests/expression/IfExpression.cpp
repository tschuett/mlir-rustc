#include "AST/IfExpression.h"

#include "ADT/CanonicalPath.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"
#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(IfExpressionTest, CheckIfExpression) {

  std::string text = "if 5 { 4; }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 7;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(IfExpressionTest, CheckIfExpression1) {

  std::string text = "if 5 { 4; } else { 5; }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 12;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(IfExpressionTest, CheckIfExpression2) {

  std::string text = "if 5 { 4; } else if true { 5; }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 14;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(IfExpressionTest, CheckIfExpression3) {

  std::string text = "if 5 { 4; } else if true { 5; } else { 6; }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 19;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(IfExpressionTest, CheckIfExpression4) {

  std::string text = "if 5 { 4; } else if true { 5; } if true { 6; } else { 6; }";

  TokenStream ts = lex(text, "lib.rs");

  printTokenState(ts.getAsView());

  size_t expectedLendth = 25;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseIfExpression({});

  EXPECT_TRUE(result.isOk());
};
