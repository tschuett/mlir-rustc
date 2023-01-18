#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(PatternTupleTest, CheckTuplePattern1) {

  std::string text = "(mut v, w)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>>
      pattern = parser.tryParseTuplePattern(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}

TEST(PatternTupleTest, CheckTuplePattern2) {

  std::string text = R"del(10, "ten")del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>>
      pattern = parser.tryParseTuplePattern(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}
