#include "Opts.inc"
#include "Rustc.h"
#include "Toml/Toml.h"

#include <fstream>
#include <llvm/ADT/StringRef.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <string>

const std::string PATH = "/Users/schuett/Work/aws_ec2_analyzer/Cargo.toml";

using namespace llvm;
using namespace rust_compiler;
using namespace rust_compiler::toml;
using namespace rust_compiler::minicargo;

namespace {
using namespace llvm::opt;
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) llvm::StringLiteral NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

const llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class MiniCargoOptTable : public llvm::opt::GenericOptTable {
public:
  MiniCargoOptTable() : opt::GenericOptTable(InfoTable) {}
};

} // namespace

int main(int argc, char **argv) {
  MiniCargoOptTable Tbl;
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver{A};
  llvm::opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](llvm::StringRef Msg) {
        llvm::errs() << Msg << '\n';
        std::exit(1);
      });

  std::ifstream t(PATH);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string file = buffer.str();

  std::optional<Toml> toml = readToml(file);
  if (not toml) {
    return 1;
  }

  printf("found toml\n");

  invokeRustC(*toml);
  return 0;
}
