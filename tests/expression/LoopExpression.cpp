#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(LoopExpressionTest, CheckLoopExpr1) {

  std::string text = "while i < 10 {i = i + 1;}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
    parser.parsePredicateLoopExpression({});

  EXPECT_TRUE(result.isOk());
};

//TEST(LoopExpressionTest, CheckLoopExpr2) {
//
//  std::string text = "let mut i = 0; while i < 10 {i = i + 1;}";
//
//  TokenStream ts = lex(text, "lib.rs");
//
//  Parser parser = {ts};
//
//  Result<std::shared_ptr<rust_compiler::ast::Expression>, std::string> result =
//    parser.parsePredicateLoopExpression({});
//
//  EXPECT_TRUE(result.isOk());
//};
