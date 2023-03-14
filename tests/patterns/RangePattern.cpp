#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(RangePatternTest, CheckRangePattern5) {
  std::string text = "-1...-2";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseRangePattern();

  if (!result)
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}

TEST(RangePatternTest, CheckRangePattern4) {
  std::string text = "1...2";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseRangePattern();

  EXPECT_TRUE(result.isOk());
}

TEST(RangePatternTest, CheckRangePattern3) {
  std::string text = "1..";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseRangePattern();

  EXPECT_TRUE(result.isOk());
}

TEST(RangePatternTest, CheckRangePattern2) {
  std::string text = "..= 2";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseRangePattern();

  EXPECT_TRUE(result.isOk());
}

TEST(RangePatternTest, CheckRangePattern1) {
  std::string text = "1 ..= 2";

  TokenStream ts = lex(text, "lib.rs");

  //ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseRangePattern();

  EXPECT_TRUE(result.isOk());
}
