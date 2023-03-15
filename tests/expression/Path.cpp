#include "AST/Crate.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

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
