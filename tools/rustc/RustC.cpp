#include "Basic/Edition.h"
#include "CrateBuilder.h"
#include "CrateLoader/CrateLoader.h"
#include "Frontend/CompilerInstance.h"
#include "Frontend/CompilerInvocation.h"
#include "Frontend/FrontendActions.h"
#include "Toml/Toml.h"

#include <fstream>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <string>

using namespace llvm;
using namespace rust_compiler::frontend;
using namespace rust_compiler;
using namespace rust_compiler::toml;
using namespace rust_compiler::crate_loader;

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

class RustCOptTable : public llvm::opt::GenericOptTable {
public:
  RustCOptTable() : opt::GenericOptTable(InfoTable) {}
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM x(argc, argv);
  RustCOptTable tbl;
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

  std::optional<std::string> crateName = std::nullopt;
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_crate_EQ)) {
    crateName = A->getValue();
  } else {
    errs() << "the crate name is missing"
           << "\n";
    exit(EXIT_FAILURE);
  }

  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_syntaxonly)) {
    CompilerInstance instance;
    SyntaxOnlyAction action;

    action.setInstance(&instance);
    action.setCurrentInput();
    action.setEdition(basic::Edition::Edition2024);

    action.run();

  } else if (const llvm::opt::Arg *A = Args.getLastArg(OPT_withsema)) {
    CompilerInstance instance;
    WithSemaAction action;

    action.setInstance(&instance);
    action.setCurrentInput();
    action.setEdition(basic::Edition::Edition2024);

    action.run();
  } else if (const llvm::opt::Arg *A = Args.getLastArg(OPT_compile)) {
    CompilerInstance instance;
    CodeGenAction action;

    action.setInstance(&instance);
    action.setCurrentInput();
    action.setEdition(basic::Edition::Edition2024);

    action.run();
  } else {
    // error
  }

//  rust_compiler::rustc::buildCrate(*path, *crateName, 1,
//                                   basic::Edition::Edition2024, mode);
}
