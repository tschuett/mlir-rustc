#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(Match2Test, CheckMatch7) {
  std::string text = R"del(
fn foo() {
    match v[..] {
        [a, b, c] => {  }
    };
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(Match2Test, CheckMatch6) {
  std::string text = R"del(
fn foo() {
    match v {
        [a, b] => {  }
        [a, b, c] => {  }
    };
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(Match2Test, CheckMatch5) {
  std::string text = R"del(
fn foo() {
    match v[..] {
        [a, b] => {  }
        [a, b, c] => {  }
    };
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(Match2Test, CheckMatch4) {
  std::string text = R"del(
fn foo() {
    match v[..] {
        [a, b] => {  }
        [a, b, c] => {  }
        _ => {  }
    };
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(Match2Test, CheckMatch3) {
  std::string text = R"del(
fn foo() {
    let int_reference = &3;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(Match2Test, CheckMatch2) {
  std::string text = R"del(
fn foo() {
    let int_reference = &3;
    match int_reference {
        &(0..=5) => (),
        _ => (),
    }
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(Match2Test, CheckMatch1) {
  std::string text = R"del(
fn foo() {
    let int_reference = &3;
    match int_reference {
        &(0..=5) => (),
        _ => (),
    }

    let arr = [1, 2, 3];
    match arr {
        [1, _, _] => "starts with one",
        [a, b, c] => "starts with something else",
    };

    // Dynamic size
    match v[..] {
        [a, b] => { /* this arm will not apply because the length doesnt match */ }
        [a, b, c] => { /* this arm will apply */ }
        _ => { /* this wildcard is required, since the length is not known statically */ }
    };
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};
