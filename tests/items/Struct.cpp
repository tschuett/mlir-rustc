#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"

#include <gtest/gtest.h>

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(StructTest, CheckStruct2) {

  std::string text = "struct Struct {field: i32}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseStruct(std::nullopt);

  if (!result)
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(StructTest, CheckStruct1) {

  std::string text = "struct Struct {field: i32}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  if (!result)
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};
