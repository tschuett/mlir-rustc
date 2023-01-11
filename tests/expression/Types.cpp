#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Type.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(TypeTest, CheckType1) {

  std::string text = "usize";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParseType(ts.getAsView());

  EXPECT_TRUE(type.has_value());

  size_t expectedLendth = 1;

  EXPECT_EQ((*type)->getTokens(), expectedLendth);
};

TEST(TypeTest, CheckType2) {

  std::string text = "f32";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParseType(ts.getAsView());

  EXPECT_TRUE(type.has_value());

  size_t expectedLendth = 1;

  EXPECT_EQ((*type)->getTokens(), expectedLendth);
};
