#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::parser {

void Parser::printFunctionStack() {
  llvm::outs() << "parsing failed"
               << "\n";
  while (!functionStack.empty()) {
    llvm::outs() << "    : " << functionStack.top() << "\n";
    functionStack.pop();
  }
}

  void Parser::pushFunction(std::string_view f) { functionStack.push(std::string(f)); }

void Parser::popFunction(std::string_view f) {
  assert(functionStack.top() == f);
  functionStack.pop();
}

} // namespace rust_compiler::parser
