#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "gtest/gtest.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(PatternTupleTest, CheckTuplePattern1) {

  std::string text = "(mut v, w)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTuplePattern();

  EXPECT_TRUE(result.isOk());
}

TEST(PatternTupleTest, CheckTuplePattern2) {

  std::string text = R"del((10, "ten"))del";

  TokenStream ts = lex(text, "lib.rs");

  ts.print(10);

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTuplePattern();

  if (result.isErr())
    llvm::errs() << result.getError() << "\n";

  EXPECT_TRUE(result.isOk());
}
