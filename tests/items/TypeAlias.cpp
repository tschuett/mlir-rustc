#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(TypeAliasTest, CheckAlias2) {

  std::string text = "type Binop = fn(i32, i32) -> i32;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseTypeAlias(std::nullopt);

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(TypeAliasTest, CheckAlias1) {

  std::string text = "type Binop = fn(i32, i32) -> i32;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  EXPECT_TRUE(result.isOk());
};
