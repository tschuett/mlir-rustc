#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(TypesTest, Checki128) {

  std::string text = "i128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>, std::string> type =
      parser.parseType();

  EXPECT_TRUE(type.isOk());
};

TEST(TypesTest, Checkf64) {

  std::string text = "f64";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>, std::string> type =
      parser.parseType();

  EXPECT_TRUE(type.isOk());
};

TEST(TypesTest, Checkisize) {

  std::string text = "isize";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>, std::string> type =
      parser.parseType();

  EXPECT_TRUE(type.isOk());
};

TEST(TypesTest, CheckBool) {

  std::string text = "bool";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>, std::string> type =
      parser.parseType();

  EXPECT_TRUE(type.isOk());
};

TEST(TypesTest, CheckBoolAsType) {

  std::string text = "bool";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>, std::string> type =
      parser.parseType();

  EXPECT_TRUE(type.isOk());
};

TEST(TypesTest, CheckI128AsType) {

  std::string text = "i128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>, std::string> type =
      parser.parseType();

  EXPECT_TRUE(type.isOk());
};
