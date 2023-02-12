#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(BorrowExpressionTest, CheckBorrowExpr1) {

  std::string text = "&foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> borrow =
      parser.tryParseBorrowExpression(ts.getAsView());

  EXPECT_TRUE(borrow.has_value());
};

TEST(BorrowExpressionTest, CheckBorrowExpr2) {

  std::string text = "&&foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> borrow =
      parser.tryParseBorrowExpression(ts.getAsView());

  EXPECT_TRUE(borrow.has_value());
};

TEST(BorrowExpressionTest, CheckBorrowExpr3) {

  std::string text = "&&mut foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> borrow =
      parser.tryParseBorrowExpression(ts.getAsView());

  EXPECT_TRUE(borrow.has_value());
};

TEST(BorrowExpressionTest, CheckBorrowExpr4) {

  std::string text = "&mut foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<std::shared_ptr<rust_compiler::ast::Expression>> borrow =
      parser.tryParseBorrowExpression(ts.getAsView());

  EXPECT_TRUE(borrow.has_value());
};
