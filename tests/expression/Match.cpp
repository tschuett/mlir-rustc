#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(MatchTest, CheckMatch8) {
  std::string text = R"del(
fn main() {
  match s{
      Point{x : 10, y : 20} => (),
      Point{y : 10, x : 20} => (),
      Point{x : 10, z : 10} => (),
  }
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

