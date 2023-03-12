#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

#include <gtest/gtest.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::parser;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

TEST(AxumTest, CheckPattern2) {
  std::string text = "OriginalUri(original_uri)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTupleStructPattern();

  EXPECT_TRUE(result.isOk());
};

TEST(AxumTest, CheckPattern1) {
  std::string text = "State(state)";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
         std::string>
      result = parser.parseTupleStructPattern();

  EXPECT_TRUE(result.isOk());
};

TEST(AxumTest, CheckType1) {
  std::string text = "State<Arc<GitHuBCIState>>";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::types::TypeExpression>,
         std::string>
      result = parser.parseType();

  EXPECT_TRUE(result.isOk());
};

TEST(AxumTest, CheckFunctionParameters1) {

  std::string text =
      "State(state): State<Arc<GitHuBCIState>>, OriginalUri(original_uri): "
      "OriginalUri, headers: HeaderMap,";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<rust_compiler::ast::FunctionParameters, std::string> result =
      parser.parseFunctionParameters();

  EXPECT_TRUE(result.isOk());
};

TEST(AxumTest, CheckFunction2) {
  std::string text = "async fn root(State(state): State<Arc<GitHuBCIState>>, "
                     "OriginalUri(original_uri): OriginalUri, headers: "
                     "HeaderMap,) -> impl IntoResponse {}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseFunction(std::nullopt);

  EXPECT_TRUE(result.isOk());
};

TEST(AxumTest, CheckFunction1) {
  std::string text = "async fn root(State(state): State<Arc<GitHuBCIState>>, "
                     "OriginalUri(original_uri): OriginalUri, headers: "
                     "HeaderMap,) -> impl IntoResponse {}";

  TokenStream ts = lex(text, "lib.rs");

  Parser parser = {ts};

  Result<std::shared_ptr<rust_compiler::ast::Item>, std::string> result =
      parser.parseItem();

  EXPECT_TRUE(result.isOk());
};
