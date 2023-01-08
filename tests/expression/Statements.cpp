#include "AST/Statements.h"

#include "BlockExpression.h"
#include "Lexer/Lexer.h"
#include "Statement.h"
#include "Statements.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(StatementsTest, CheckStatements1) {

  std::string text = "{return left + right;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> stmts =
      tryParseStatements(ts.getAsView());

  EXPECT_TRUE(stmts.has_value());
};

TEST(StatementsTest, CheckStatements2) {

  std::string text = "return left + right;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::Statements>> stmts =
      tryParseStatements(ts.getAsView());

  EXPECT_TRUE(stmts.has_value());
};
