#include "Lexer/Lexer.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;

TEST(LexerTest, CheckKeyWord) {

  std::string text = "for async\n";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);
};

TEST(LexerTest, CheckUse) {
  std::string text = "use "
                     "aws_sdk_ec2::{error::DescribeInstanceTypesError,model::{"
                     "InstanceType, InstanceTypeInfo}, types::SdkError,};\n";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 23;

  EXPECT_EQ(ts.getLength(), expectedLendth);
};

TEST(LexerTest, CheckInteger) {
  std::string text = "128";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::INTEGER_LITERAL);
};

TEST(LexerTest, CheckDecInteger) {
  std::string text = "1";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::INTEGER_LITERAL);
};

TEST(LexerTest, CheckDollarCrate) {
  std::string text = "$crate";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  ts.print(5);

  EXPECT_EQ(ts.getLength(), expectedLendth);

  // EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::DecIntegerLiteral);
};

TEST(LexerTest, CheckKeyword) {
  std::string text = "return";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  if (KeyWord2String(ts.getAsView().front().getKeyWordKind())) {
    printf("keyword: %s\n",
           (*KeyWord2String(ts.getAsView().front().getKeyWordKind())).c_str());
  }

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Keyword);
  EXPECT_EQ(ts.getAsView().front().getKeyWordKind(), KeyWordKind::KW_RETURN);
};

TEST(LexerTest, CheckIf) {
  std::string text = "if true { 5 } else { 6 }";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 10;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView()[3].getKind(), TokenKind::INTEGER_LITERAL);
  EXPECT_EQ(ts.getAsView()[7].getKind(), TokenKind::INTEGER_LITERAL);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Keyword);
  EXPECT_EQ(ts.getAsView().front().getKeyWordKind(), KeyWordKind::KW_IF);
};

TEST(LexerTest, CheckIf1) {
  std::string text = "if true { 5 }";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 6;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView()[3].getKind(), TokenKind::INTEGER_LITERAL);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Keyword);
  EXPECT_EQ(ts.getAsView().front().getKeyWordKind(), KeyWordKind::KW_IF);
};

TEST(LexerTest, CheckFive) {
  std::string text = "5 ";

  TokenStream ts = lex(text, "lib.rs");

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::INTEGER_LITERAL);
};

TEST(LexerTest, CheckTen) {
  std::string text = "10";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::INTEGER_LITERAL);
};

TEST(LexerTest, CheckHundred) {
  std::string text = "100";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::INTEGER_LITERAL);
};

TEST(LexerTest, CheckThousand) {
  std::string text = "1000";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::INTEGER_LITERAL);
};

TEST(LexerTest, CheckLoop) {
  std::string text = "while i < 10 {i = i + 1; }";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 13;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::Keyword);
};

TEST(LexerTest, CheckAnd) {
  std::string text = "&";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 2;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::And);
};

TEST(LexerTest, CheckAndAnd) {
  std::string text = "&& && ";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  size_t expectedLendth = 3;

  EXPECT_EQ(ts.getLength(), expectedLendth);

  EXPECT_EQ(ts.getAsView().front().getKind(), TokenKind::AndAnd);
};
