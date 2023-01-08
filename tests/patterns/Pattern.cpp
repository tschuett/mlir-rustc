#include "IdentifierPattern.h"
#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(PatternTest, CheckIdentifierPattern1) {

  std::string text = "ref mut foo";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::PatternWithoutRange>>
      pattern = tryParseIdentifierPattern(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
};

TEST(PatternTest, CheckIdentifierPattern2) {

  std::string text = "ref foo";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::PatternWithoutRange>>
      pattern = tryParseIdentifierPattern(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
};

TEST(PatternTest, CheckIdentifierPattern3) {

  std::string text = "foo";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::PatternWithoutRange>>
      pattern = tryParseIdentifierPattern(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
};
