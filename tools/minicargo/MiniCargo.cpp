#include "Opts.inc"
#include "Toml/Toml.h"

#include <fstream>
#include <sstream>
#include <string>

const std::string PATH = "/Users/schuett/Work/aws_ec2_analyzer/Cargo.toml";

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

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
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
  ScanDepsOptTable() : OptTable(InfoTable) { setGroupedShortOptions(true); }
};

} // namespace

int main(int argc, char **argv) {

  std::ifstream t(PATH);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string file = buffer.str();

  std::optional<Toml> toml = readToml(file);
  if (not toml) {
    return 1;
  }

  return 0;
}
