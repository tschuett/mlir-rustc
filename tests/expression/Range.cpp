#include "AST/Expression.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(RangeTest, CheckRange7) {
  std::string text = R"del(
fn foo() {
    1..2;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(RangeTest, CheckRange6) {
  std::string text = R"del(
fn foo() {
    1..2;
    3..;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  if (result.isErr())
    llvm::outs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(RangeTest, CheckRange5) {
  std::string text = R"del(
fn foo() {
    1..2;
    3..;
    ..4;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  if (result.isErr())
    llvm::outs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(RangeTest, CheckRange4) {
  std::string text = R"del(
fn foo() {
    1..2;
    3..;
    ..4;
    ..;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  if (result.isErr())
    llvm::outs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(RangeTest, CheckRange3) {
  std::string text = R"del(
fn foo() {
    1..2;
    3..;
    ..4;
    ..;
    5..=6;
    ..=7;

    let x = std::ops::Range { start: 0, end: 10 };
    let y = 0..10;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  if (result.isErr())
    llvm::outs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(RangeTest, CheckRange2) {
  std::string text = R"del(
fn foo() {
    1..2; // std::ops::Range
    3..; // std::ops::RangeFrom
    ..4; // std::ops::RangeTo
    ..; // std::ops::RangeFull
    5..=6; // std::ops::RangeInclusive
    ..=7; // std::ops::RangeToInclusive

    let x = std::ops::Range { start: 0, end: 10 };
    let y = 0..10;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  if (result.isErr())
    llvm::outs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
};

TEST(RangeTest, CheckRange1) {
  std::string text = R"del(
fn foo() {
    1..2; // std::ops::Range
    3..; // std::ops::RangeFrom
    ..4; // std::ops::RangeTo
    ..; // std::ops::RangeFull
    5..=6; // std::ops::RangeInclusive
    ..=7; // std::ops::RangeToInclusive

    let x = std::ops::Range { start: 0, end: 10 };
    let y = 0..10;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  EXPECT_TRUE(result.isOk());
};
