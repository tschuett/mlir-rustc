#include "AST/Statement.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ArrayTest, CheckArray6) {

  std::string text = "[EMPTY; 2]";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parseArrayExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ArrayTest, CheckArray5) {

  std::string text = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parseArrayExpression({});

  EXPECT_TRUE(result.isOk());
};

//TEST(ArrayTest, CheckArray4) {
//
//  std::string text = "[0u8, 0u8, 0u8, 0u8]";
//
//  TokenStream ts = lex(text, "lib.rs");
//
//  Parser parser = {ts};
//
//  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
//    parser.parseArrayExpression({});
//
//  EXPECT_TRUE(result.isOk());
//};

TEST(ArrayTest, CheckArray3) {

  std::string text = R"foo([0; 128])foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parseArrayExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ArrayTest, CheckArray2) {

  std::string text = R"foo(["a", "b", "c", "d"])foo";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parseArrayExpression({});

  EXPECT_TRUE(result.isOk());
};

TEST(ArrayTest, CheckArray1) {

  std::string text = "[1, 2, 3, 4]";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parseArrayExpression({});

  EXPECT_TRUE(result.isOk());
};
