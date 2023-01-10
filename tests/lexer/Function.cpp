#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;

TEST(ArithmeticLexerTest, CheckPlus) {
  std::string text = "+";

  TokenStream ts =
      lex(text, "pub fn add(right: usize) -> usize {return right + right;}");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Plus);
};
