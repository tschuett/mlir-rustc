#include "Frontend/FrontendAction.h"

#include "Basic/Ids.h"
#include "CrateLoader/CrateLoader.h"
#include "Frontend/FrontendOptions.h"
#include "Sema/Sema.h"

using namespace rust_compiler::sema;
using namespace rust_compiler::crate_loader;

namespace rust_compiler::frontend {

bool FrontendAction::runParse() {
  basic::CrateNum crateNum = 1;

  switch (currentInput.getKind()) {
  case InputKind::File: {
    crate = loadCrate(currentInput.getInputFile(), currentInput.getCrateName(),
                      crateNum, LoadMode::File);
    break;
  }
  case InputKind::CargoTomlDir: {
    crate = loadCrate(currentInput.getInputFile(), currentInput.getCrateName(),
                      crateNum, LoadMode::CargoTomlDir);
    break;
  }
  }

  return true;
}

bool FrontendAction::runSemanticChecks() {
  Sema sema;
  sema.analyze(crate);

  return true;
}

} // namespace rust_compiler::frontend

// FIXME: pcms
