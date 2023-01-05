#include "Attributes.h"
#include "Lexer/Lexer.h"
#include "UseDeclaration.h"
#include "Modules.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(TypesTest, CheckClippy) {

  std::string text = "#![deny(clippy::as_conversions,clippy::missing_safety_"
                     "doc,clippy::undocumented_unsafe_blocks)]\n";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<ClippyAttribute> clippy =
      tryParseClippyAttribute(ts.getAsView());

  // size_t expectedLendth = 2;
  EXPECT_TRUE(clippy.has_value());

  //  EXPECT_EQ(ts.getLength(), expectedLendth);
};

TEST(TypesTest, CheckModuleDecl) {

  std::string text =
      "mod print_spot_region {pub mod print_spot_regions;mod "
      "data_collector;mod spot_region;mod reorder;mod printer;}\n";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Module> module = tryParseModule(ts.getAsView(), "");

  EXPECT_TRUE(module.has_value());
};
