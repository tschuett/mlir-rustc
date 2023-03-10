#include "Parser/Parser.h"

#include "Lexer/Lexer.h"

#include <gtest/gtest.h>

#include "ADT/Result.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;

TEST(ParserTest, CheckModuleDecl) {

  std::string text = "Option::Some(3)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  ts.print(10);
  
  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
      parser.parseStructExprTuple();

  EXPECT_TRUE(result.isOk());
};
