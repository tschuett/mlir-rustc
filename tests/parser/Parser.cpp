#include "Attributes.h"
#include "Lexer/Lexer.h"
#include "UseDeclaration.h"
#include "Modules.h"
#include "gtest/gtest.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;

TEST(ParserTest, CheckClippy) {

  std::string text = "#![deny(clippy::as_conversions,clippy::missing_safety_"
                     "doc,clippy::undocumented_unsafe_blocks)]\n";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<ClippyAttribute> clippy =
      tryParseClippyAttribute(ts.getAsView());

  // size_t expectedLendth = 2;
  EXPECT_TRUE(clippy.has_value());

  //  EXPECT_EQ(ts.getLength(), expectedLendth);
};

TEST(ParserTest, CheckModuleDecl) {

  std::string text =
      "mod print_spot_region {pub mod print_spot_regions;mod "
      "data_collector;mod spot_region;mod reorder;mod printer;}\n";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Module> module = tryParseModule(ts.getAsView(), "");

  EXPECT_TRUE(module.has_value());
};

TEST(ParserTest, CheckSimpleModDecl) {

  std::string text = "mod pricing;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<rust_compiler::ast::Module> module = tryParseModule(ts.getAsView(), "");

  EXPECT_TRUE(module.has_value());
};


TEST(ParserTest, CheckUseDecl) {

  std::string text = "use aws_sdk_ec2::error::DescribeSpotPriceHistoryError;";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<UseDeclaration> use = tryParseUseDeclaration(ts.getAsView());

  EXPECT_TRUE(use.has_value());
};


TEST(ParserTest, CheckSimpleUseTreeDecl) {

  std::string text = "use aws_sdk_ec2::{error::DescribeInstanceTypesError,types::SdkError,};";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<UseDeclaration> use = tryParseUseDeclaration(ts.getAsView());

  EXPECT_TRUE(use.has_value());
};

TEST(ParserTest, CheckUseTreeDecl) {

  std::string text = "use aws_sdk_ec2::{error::DescribeInstanceTypesError,model::{InstanceType, InstanceTypeInfo},types::SdkError,};";

  TokenStream ts = lex(text, "lib.rs");

  std::optional<UseDeclaration> use = tryParseUseDeclaration(ts.getAsView());

  EXPECT_TRUE(use.has_value());
};

//TEST(ParserTest, CheckUseTree2Decl) {
//
//  std::string text = "use aws_sdk_ec2::{error::DescribeInstanceTypesError,model::{InstanceType, InstanceTypeInfo}};";
//
//  TokenStream ts = lex(text, "lib.rs");
//
//  std::optional<UseDeclaration> use = tryParseUseDeclaration(ts.getAsView());
//
//  EXPECT_TRUE(use.has_value());
//};



