#include "Toml/Toml.h"

#include "llvm/Support/TargetSelect.h"

#include "Lexer.h"
#include "Parser.h"

#include <fstream>
#include <sstream>
#include <string>

using namespace rust_compiler;
using namespace rust_compiler::toml;

const std::string PATH = "/Users/schuett/Work/aws_ec2_analyzer/src/lib.rs";

int main(int argc, char **argv) {
  std::ifstream t(PATH);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string file = buffer.str();

  TokenStream ts = lex(file);
  parser(ts);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
}
