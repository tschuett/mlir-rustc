#include "PatternNoTopAlt.h"

#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(PatternNoTopAltTest, CheckIdentifierPattern1) {

  std::string text = "(mut v, w)";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::PatternNoTopAlt>> pattern =
      tryParsePatternNoTopAlt(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}
