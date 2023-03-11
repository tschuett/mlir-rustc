#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(SimplePathTest, CheckSimplePath4) {

  std::string text = "five::super::$crate";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  StringResult<rust_compiler::ast::SimplePath> result =
      parser.parseSimplePath();

  EXPECT_TRUE(result.isOk());
};

TEST(SimplePathTest, CheckSimplePath3) {

  std::string text = "::five::super::$crate";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  StringResult<rust_compiler::ast::SimplePath> result =
      parser.parseSimplePath();

  EXPECT_TRUE(result.isOk());
};

TEST(SimplePathTest, CheckSimplePath2) {

  std::string text = "::five::super";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  StringResult<rust_compiler::ast::SimplePath> result =
      parser.parseSimplePath();

  EXPECT_TRUE(result.isOk());
};

TEST(SimplePathTest, CheckSimplePath1) {

  std::string text = "::five";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  StringResult<rust_compiler::ast::SimplePath> result =
      parser.parseSimplePath();

  EXPECT_TRUE(result.isOk());
};

