#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;

TEST(ArithmeticLexerTest, CheckPlus) {
  std::string text = "+";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Plus);
};


TEST(ArithmeticLexerTest, CheckMinus) {
  std::string text = "-";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Minus);
};


TEST(ArithmeticLexerTest, CheckStar) {
  std::string text = "*";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Star);
};


TEST(ArithmeticLexerTest, CheckSlash) {
  std::string text = "/";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Slash);
};

TEST(ArithmeticLexerTest, CheckPercent) {
  std::string text = "%";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Percent);
};

TEST(ArithmeticLexerTest, CheckAnd) {
  std::string text = "&";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::And);
};

TEST(ArithmeticLexerTest, CheckOr) {
  std::string text = "|";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Or);
};

TEST(ArithmeticLexerTest, CheckCaret) {
  std::string text = "^";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Caret);
};

TEST(ArithmeticLexerTest, CheckShl) {
  std::string text = "<<";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Shl);
};

TEST(ArithmeticLexerTest, CheckShr) {
  std::string text = ">>";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 1;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Shr);
};
