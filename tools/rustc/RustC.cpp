#include "Basic/Edition.h"
#include "CrateBuilder.h"
#include "CrateLoader/CrateLoader.h"
#include "Frontend/CompilerInstance.h"
#include "Frontend/CompilerInvocation.h"
#include "Frontend/FrontendActions.h"
#include "Frontend/FrontendOptions.h"
#include "Toml/Toml.h"

#include <fstream>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
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

  std::string path;
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_path_EQ)) {
    path = A->getValue();
  } else {
    errs() << "the input path is missing"
           << "\n";
    exit(EXIT_FAILURE);
  }

  std::string crateName = "";
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_crate_EQ)) {
    crateName = A->getValue();
  } else {
    errs() << "the crate name is missing"
           << "\n";
    exit(EXIT_FAILURE);
  }

  std::string remarksOutput;
  llvm::SmallVector<char, 128> libFile{path.begin(), path.end()};
  llvm::sys::path::replace_extension(libFile, ".yaml");
  remarksOutput = {libFile.begin(), libFile.end()};

  CompilerInstance instance;
  FrontendInput input = {path, remarksOutput, crateName, InputKind::File};

  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_syntaxonly)) {
    SyntaxOnlyAction action;

    action.setInstance(&instance);
    action.setCurrentInput(input);
    action.setEdition(basic::Edition::Edition2024);

    action.execute();

  } else if (const llvm::opt::Arg *A = Args.getLastArg(OPT_withsema)) {
    SemaOnlyAction action;

    action.setInstance(&instance);
    action.setCurrentInput(input);
    action.setEdition(basic::Edition::Edition2024);

    action.execute();
  } else if (const llvm::opt::Arg *A = Args.getLastArg(OPT_compile)) {
    CodeGenAction action;

    action.setInstance(&instance);
    action.setCurrentInput(input);
    action.setEdition(basic::Edition::Edition2024);

    action.execute();
  } else {
    // error
  }

  //  rust_compiler::rustc::buildCrate(*path, *crateName, 1,
  //                                   basic::Edition::Edition2024, mode);
}

// FIXME InputKind::CargoTomlDir
