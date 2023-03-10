#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <string>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ExpressionWithoutBlockTest, CheckLiteralBlock) {

  std::string text = "128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restrictions;
  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseExpressionWithoutBlock({}, restrictions);

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionWithoutBlockTest, CheckPathBlock1) {

  std::string text = "local_var";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restrictions;
  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseExpressionWithoutBlock({}, restrictions);

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionWithoutBlockTest, CheckPathBlock2) {

  std::string text = "globals::STATIC_VAR;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restrictions;
  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseExpressionWithoutBlock({}, restrictions);

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionWithoutBlockTest, CheckOperatorBlock1) {

  std::string text = "1 + 2";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restrictions;
  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseExpressionWithoutBlock({}, restrictions);

  EXPECT_TRUE(result.isOk());
};

TEST(ExpressionWithoutBlockTest, CheckOperatorBlock2) {

  std::string text = "a + b";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Restrictions restrictions;
  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseExpressionWithoutBlock({}, restrictions);

  EXPECT_TRUE(result.isOk());
};
