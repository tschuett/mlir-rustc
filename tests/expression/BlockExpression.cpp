#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(BlockExpressionTest, CheckBlockExpr1) {

  std::string text = "{}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(BlockExpressionTest, CheckBlockExpr2) {

  std::string text = "{ ; }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(BlockExpressionTest, CheckBlockExpr37) {

  std::string text = "{ true; }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  if (!result)
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(BlockExpressionTest, CheckBlockExpr41) {

  std::string text = "{ super; }";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(BlockExpressionTest, CheckBlockExpr5) {

  std::string text = "{return 6 + 5;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(BlockExpressionTest, CheckBlockExpr40) {

  std::string text = "{return left + right;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseBlockExpression({});

  EXPECT_TRUE(result.isOk());
};
