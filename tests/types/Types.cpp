#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "PrimitiveType.h"
#include "Type.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(TypesTest, Checki128) {

  std::string text = "i128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParsePrimitiveType(ts.getAsView());

  EXPECT_TRUE(type.has_value());
};

TEST(TypesTest, Checkf64) {

  std::string text = "f64";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParsePrimitiveType(ts.getAsView());

  EXPECT_TRUE(type.has_value());
};

TEST(TypesTest, Checkisize) {

  std::string text = "isize";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParsePrimitiveType(ts.getAsView());

  EXPECT_TRUE(type.has_value());
};

TEST(TypesTest, CheckBool) {

  std::string text = "bool";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParsePrimitiveType(ts.getAsView());

  EXPECT_TRUE(type.has_value());
};

TEST(TypesTest, CheckBoolAsType) {

  std::string text = "bool";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParseType(ts.getAsView());

  EXPECT_TRUE(type.has_value());
};

TEST(TypesTest, CheckI128AsType) {

  std::string text = "i128";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts, ""};

  std::optional<std::shared_ptr<rust_compiler::ast::types::Type>> type =
      parser.tryParseType(ts.getAsView());

  EXPECT_TRUE(type.has_value());
};
