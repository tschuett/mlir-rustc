#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::patterns;

TEST(PatternTest, CheckIdentifierPattern1) {

  std::string text = "ref mut foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>, std::string>
      result = parser.parseIdentifierPattern();


  EXPECT_TRUE(result.isOk());
};

TEST(PatternTest, CheckIdentifierPattern2) {

  std::string text = "ref foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>, std::string>
      result = parser.parseIdentifierPattern();


  EXPECT_TRUE(result.isOk());
};

TEST(PatternTest, CheckIdentifierPattern3) {

  std::string text = "foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>, std::string>
      result = parser.parseIdentifierPattern();

  EXPECT_TRUE(result.isOk());
};
