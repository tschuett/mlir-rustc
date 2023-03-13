#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(StructPatternTest, CheckStructPattern5) {

  std::string text = "Point{x: _}";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseStructPattern();

  EXPECT_TRUE(result.isOk());
}

TEST(StructPatternTest, CheckStructPattern4) {

  std::string text = "Point{..}";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseStructPattern();

  EXPECT_TRUE(result.isOk());
}

TEST(StructPatternTest, CheckStructPattern3) {

  std::string text = "Point{x : 10, y : 20,}";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseStructPattern();

  EXPECT_TRUE(result.isOk());
}

TEST(StructPatternTest, CheckStructPattern2) {

  std::string text = "Point{x : 10, y : 20, ..}";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseStructPattern();

  EXPECT_TRUE(result.isOk());
}

TEST(StructPatternTest, CheckStructPattern1) {

  std::string text = "Point{x : 10, y : 20}";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseStructPattern();

  EXPECT_TRUE(result.isOk());
}
