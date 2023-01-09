#include "Statement.h"

#include "ExpressionWithoutBlock.h"

#include "ReturnExpression.h"
#include "ExpressionStatement.h"
#include "BlockExpression.h"
#include "Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(BlockExpressionTest, CheckBlockExpr1) {

  std::string text = "{}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};

TEST(BlockExpressionTest, CheckBlockExpr2) {

  std::string text = "{ ; }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};

TEST(BlockExpressionTest, CheckBlockExpr3) {

  std::string text = "{ true }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};


TEST(BlockExpressionTest, CheckBlockExpr4) {

  std::string text = "{ super }";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};

TEST(BlockExpressionTest, CheckBlockExpr5) {

  std::string text = "{return 6 + 5;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};


TEST(BlockExpressionTest, CheckBlockExpr30) {

  std::string text = "{return left + right;}";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<std::shared_ptr<rust_compiler::ast::BlockExpression>> block =
      tryParseBlockExpression(ts.getAsView());

  EXPECT_TRUE(block.has_value());
};

