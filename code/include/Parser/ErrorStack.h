#pragma once

#include <string_view>

namespace rust_compiler::parser {

class Parser;

class ParserErrorStack {
  Parser *parser;
  std::string functionName;
public:
  ParserErrorStack(Parser *parser, std::string_view functionName);
  ~ParserErrorStack();
};

}; // namespace rust_compiler::parser
