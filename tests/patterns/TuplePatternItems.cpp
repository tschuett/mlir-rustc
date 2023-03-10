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

  Parser parser = {ts};

  Result<rust_compiler::ast::patterns::TuplePatternItems, std::string> result =
      parser.parseTuplePatternItems();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems2) {

  std::string text = "..";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::patterns::TuplePatternItems, std::string> result =
      parser.parseTuplePatternItems();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems3) {

  std::string text = "5,";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::patterns::TuplePatternItems, std::string> result =
      parser.parseTuplePatternItems();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems4) {

  std::string text = "5,5,";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::patterns::TuplePatternItems, std::string> result =
      parser.parseTuplePatternItems();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternTupleItemsTest, CheckTuplePatternItems5) {

  std::string text = "5,5";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::patterns::TuplePatternItems, std::string> result =
      parser.parseTuplePatternItems();

  EXPECT_TRUE(result.isOk());
}
