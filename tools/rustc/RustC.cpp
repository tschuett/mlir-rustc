#include "CrateBuilder.h"
#include "Toml/Toml.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InitLLVM.h"

#include <fstream>
#include <sstream>
#include <string>

using namespace rust_compiler;
using namespace rust_compiler::toml;

namespace {
using namespace llvm::opt;
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) llvm::StringLiteral NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

const llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {PREFIX,      NAME,      HELPTEXT,                                           \
   METAVAR,     OPT_##ID,  llvm::opt::Option::KIND##Class,                     \
   PARAM,       FLAGS,     OPT_##GROUP,                                        \
   OPT_##ALIAS, ALIASARGS, VALUES},
#include "Opts.inc"
#undef OPTION
};

class MiniCargoOptTable : public llvm::opt::OptTable {
public:
  MiniCargoOptTable() : OptTable(InfoTable) { setGroupedShortOptions(true); }
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM x(argc, argv);
  MiniCargoOptTable tbl;
  llvm::StringRef ToolName = argv[0];
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver{A};
  llvm::opt::InputArgList Args =
      tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](llvm::StringRef Msg) {
        llvm::errs() << Msg << '\n';
        std::exit(1);
      });

  // Initialize targets first, so that --version shows registered targets.
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  if (Args.hasArg(OPT_help)) {
    tbl.printHelp(llvm::outs(), "rustc [options]", "rustc");
    std::exit(0);
  }

  if (Args.hasArg(OPT_version)) {
    llvm::outs() << ToolName << " 0.9" << '\n';
    std::exit(0);
  }

  std::string edition = "2021";
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_edition_EQ)) {
    edition = A->getValue();
  }

  std::optional<std::string> path = std::nullopt;
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_path_EQ)) {
    path = A->getValue();
  }

  if (not path) {
    llvm::errs() << "path parameter is missing" << '\n';
    return -1;
  }

  llvm::outs() << "path :" << *path << '\n';

  rust_compiler::rustc::buildCrate(*path, edition);
}
