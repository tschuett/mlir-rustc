#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(MatchTest, CheckMatch1) {
  std::string text = R"del(
fn main() {
  match s{
      Point{x : 10, y : 20} = > (),
      Point{y : 10, x : 20} = > (), // order doesn't matter
      Point{x : 10, ..} = > (),
      Point{..} = > (),
  }

  match t {
    PointTuple{0 : 10, 1 : 20} = > (),
    PointTuple{1 : 10, 0 : 20} = > (), // order doesn't matter
    PointTuple{0 : 10, ..} = > (), PointTuple{..} = > (),
  }
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>,
         std::string>
    result = parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};
