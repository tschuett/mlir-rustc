#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Sema/Sema.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::sema;

TEST(SemaTest, CheckSema1) {
  std::string text = R"del(
fn foo(a: i32, b: i32) -> i32 {
    return a + b;
}
)del";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Crate>, std::string> result =
    parser.parseCrateModule("crate", 5);

  if (!result)
    llvm::errs() << "error: " << result.getError() << "/n";

  EXPECT_TRUE(result.isOk());

  std::shared_ptr<rust_compiler::ast::Crate> crate = result.getValue();

  Sema sema;
  sema.analyze(crate);
};
