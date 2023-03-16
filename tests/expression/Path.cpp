#include "AST/Crate.h"
#include "AST/Function.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

// PathInExpression
TEST(PathTest, CheckPath8) {
  std::string text = R"del(
fn bar() {
    Vec::<u8>::with_capacity(1024);
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Crate>, std::string> result =
      parser.parseCrateModule("name", 5);

  EXPECT_TRUE(result.isOk());
};

// MethodCallExpression
TEST(PathTest, CheckPath7) {
  std::string text = R"del(
fn bar() {
    (0..10).collect::<Vec<_>>();
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Crate>, std::string> result =
      parser.parseCrateModule("name", 5);

  EXPECT_TRUE(result.isOk());
};

TEST(PathTest, CheckPath6) {
  std::string text = R"del(
fn bar() {
    (0..10).collect::<Vec<_>>();
    Vec::<u8>::with_capacity(1024);
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Crate>, std::string> result =
      parser.parseCrateModule("name", 5);

  EXPECT_TRUE(result.isOk());
};

TEST(PathTest, CheckPath5) {
  std::string text = R"del(
fn foo() {
    <[i32]>::reverse;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(PathTest, CheckPath4) {
  std::string text = R"del(
fn foo() {
    let slice_reverse = <[i32]>::reverse;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(PathTest, CheckPath3) {
  std::string text = R"del(
fn foo() {
    let push_integer = Vec::<i32>::push;
    let slice_reverse = <[i32]>::reverse;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(PathTest, CheckPath2) {
  std::string text = R"del(
fn foo() {
    let some_constructor = Some::<i32>;
    let push_integer = Vec::<i32>::push;
    let slice_reverse = <[i32]>::reverse;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(PathTest, CheckPath1) {
  std::string text = R"del(
fn foo() {
    local_var;
    globals::STATIC_VAR;
    unsafe { globals::STATIC_MUT_VAR };
    let some_constructor = Some::<i32>;
    let push_integer = Vec::<i32>::push;
    let slice_reverse = <[i32]>::reverse;
}

fn bar() {
    (0..10).collect::<Vec<_>>();
    Vec::<u8>::with_capacity(1024);
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Crate>, std::string> result =
      parser.parseCrateModule("name", 5);

  EXPECT_TRUE(result.isOk());
};
