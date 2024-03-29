#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ReferencePatternTest, CheckReferencePattern1) {
  std::string text = "&[u32]";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseReferencePattern();

  EXPECT_TRUE(result.isOk());
};
