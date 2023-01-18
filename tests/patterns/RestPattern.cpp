#include "AST/Patterns/PatternNoTopAlt.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(RestPatternTest, CheckRestPattern1) {

  std::string text = "..";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>>
      pattern = parser.tryParseRestPattern(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}
