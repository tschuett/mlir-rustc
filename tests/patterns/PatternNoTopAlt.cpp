#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(PatternNoTopAltTest, CheckIdentifierPattern9) {

  std::string text = "(w, v)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTupleOrGroupedPattern();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern8) {

  std::string text = "(w, v)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTuplePattern();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern7) {

  std::string text = "w";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseIdentifierOrPathPattern();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern6) {

  std::string text = "w";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result =
          parser
              .parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern5) {

  std::string text = "w";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parsePathPattern();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern4) {

  std::string text = "w";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseIdentifierPattern();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern3) {

  std::string text = "w";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::Pattern>, std::string>
      result = parser.parsePattern();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern2) {

  std::string text = "mut v";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::Pattern>, std::string>
      result = parser.parsePattern();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern1) {

  std::string text = "(mut v, w)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTupleOrGroupedPattern();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";


  EXPECT_TRUE(result.isOk());
}

TEST(PatternNoTopAltTest, CheckIdentifierPattern0) {

  std::string text = "(mut v, w)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parsePatternNoTopAlt();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}
