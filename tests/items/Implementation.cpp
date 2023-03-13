#include "AST/AssociatedItem.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ImplementationTest, CheckImplStruct2) {

  std::string text = "impl Struct {fn new() -> Struct {Struct { field: 32 };}}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
    parser.parseImplementation({});

  EXPECT_TRUE(result.isOk());
};

TEST(ImplementationTest, CheckImplStruct1) {

  std::string text = "impl Struct {fn new() -> Struct {Struct { field: 32 };}}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  EXPECT_TRUE(result.isOk());
};
