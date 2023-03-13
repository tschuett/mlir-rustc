#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Util.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(FunctionTest, CheckFun3) {
  std::string text = R"del(
fn main() {
    let mut x = add(5, 7);

    type Binop = fn(i32, i32) -> i32;
    let bo: Binop = add;
    x = bo(5, 7);
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFun2) {
  std::string text = R"del(
fn main() {
    let mut x = add(5, 7);

    type Binop = fn(i32, i32) -> i32;
    let bo: Binop = add;
    x = bo(5, 7);
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(FunctionTest, CheckFun1) {

  std::string text = R"del(
fn main() {
    let mut x = add(5, 7);

    type Binop = fn(i32, i32) -> i32;
    let bo: Binop = add;
    x = bo(5, 7);
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  EXPECT_TRUE(result.isOk());
};




