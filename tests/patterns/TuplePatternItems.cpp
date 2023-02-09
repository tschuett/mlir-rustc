#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(PatternTupleItemsTest, CheckTuplePatternItems1) {

  std::string text = R"del(5,)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::TuplePatternItems>>
      pattern = parser.tryParseTuplePatternItems(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems2) {

  std::string text = "..";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::TuplePatternItems>>
      pattern = parser.tryParseTuplePatternItems(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems3) {

  std::string text = "5,";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::TuplePatternItems>>
      pattern = parser.tryParseTuplePatternItems(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems4) {

  std::string text = "5,5,";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::TuplePatternItems>>
      pattern = parser.tryParseTuplePatternItems(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems5) {

  std::string text = "5,5";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, CanonicalPath("")};

  std::optional<std::shared_ptr<rust_compiler::ast::patterns::TuplePatternItems>>
      pattern = parser.tryParseTuplePatternItems(ts.getAsView());

  EXPECT_TRUE(pattern.has_value());
}
