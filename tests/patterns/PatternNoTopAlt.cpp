#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(PatternNoTopAltTest, CheckIdentifierPattern1) {

  std::string text = "(mut v, w)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>>
      pattern = parser.tryParsePatternNoTopAlt(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}
