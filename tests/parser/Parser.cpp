#include "Parser/Parser.h"

#include "Lexer/Lexer.h"

#include <gtest/gtest.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(ParserTest, CheckModuleDecl) {

  std::string text =
      "mod print_spot_region {pub mod print_spot_regions;mod "
      "data_collector;mod spot_region;mod reorder;mod printer;}\n";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<rust_compiler::ast::Module> module =
      parser.tryParseModule(ts.getAsView());

  EXPECT_TRUE(module.has_value());
};

TEST(ParserTest, CheckSimpleModDecl) {

  std::string text = "mod pricing;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<rust_compiler::ast::Module> module =
      parser.tryParseModule(ts.getAsView());

  EXPECT_TRUE(module.has_value());
};

TEST(ParserTest, CheckUseDecl) {

  std::string text = "use aws_sdk_ec2::error::DescribeSpotPriceHistoryError;";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<UseDeclaration> use =
      parser.tryParseUseDeclaration(ts.getAsView());

  EXPECT_TRUE(use.has_value());
};

TEST(ParserTest, CheckSimpleUseTreeDecl) {

  std::string text =
      "use aws_sdk_ec2::{error::DescribeInstanceTypesError,types::SdkError,};";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<UseDeclaration> use =
      parser.tryParseUseDeclaration(ts.getAsView());

  EXPECT_TRUE(use.has_value());
};

TEST(ParserTest, CheckUseTreeDecl) {

  std::string text = "use "
                     "aws_sdk_ec2::{error::DescribeInstanceTypesError,model::{"
                     "InstanceType, InstanceTypeInfo},types::SdkError,};";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  std::optional<UseDeclaration> use =
      parser.tryParseUseDeclaration(ts.getAsView());

  EXPECT_TRUE(use.has_value());
};

// TEST(ParserTest, CheckUseTree2Decl) {
//
//   std::string text = "use
//   aws_sdk_ec2::{error::DescribeInstanceTypesError,model::{InstanceType,
//   InstanceTypeInfo}};";
//
//   TokenStream ts = lex(text, "lib.rs");
//
//   std::optional<UseDeclaration> use = tryParseUseDeclaration(ts.getAsView());
//
//   EXPECT_TRUE(use.has_value());
// };
