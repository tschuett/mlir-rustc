#include "Parser/Parser.h"

#include "Lexer/Lexer.h"
#include <llvm/Support/Error.h>

#include <gtest/gtest.h>
// #include <llvm/Testing/Support/Error.h>

#include "ADT/Error.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

TEST(ParserTest, CheckModuleDecl) {

  std::string text = "Option::Some(3)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  llvm::Expected<std::shared_ptr<rust_compiler::ast::Expression>> result =
      parser.parseStructExprTuple();

  if (auto E = result.takeError()) {
    std::move(E);
    EXPECT_TRUE(false);
  } else {
    EXPECT_TRUE(true);
  }

  EXPECT_THAT_EXPECTED(result, Succeeded());
  // EXPECT_TRUE(module.has_value());
};
