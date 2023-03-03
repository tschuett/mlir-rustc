#include "Parser/ErrorStack.h"

namespace rust_compiler::parser {

ParserErrorStack::ParserErrorStack(Parser *_parser,
                                   std::string_view _functionName) {
  functionName = _functionName;
  parser = _parser;
  parser->pushFunction(functionName);
}

ParserErrorStack::~ParserErrorStack() { parser->popFunction(functionName); }

} // namespace rust_compiler::parser
