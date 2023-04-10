#include "Frontend/FrontendAction.h"

#include "Basic/Ids.h"
#include "CrateLoader/CrateLoader.h"
#include "Frontend/FrontendOptions.h"
#include "Sema/Sema.h"
#include "Session/Session.h"
#include "TyCtx/TyCtx.h"

#include <llvm/Support/Error.h>
#include <random>

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

void FrontendAction::setEdition(basic::Edition _edition) { edition = _edition; }

ast::Crate *FrontendAction::getCrate() { return crate.get(); }

void FrontendAction::setCurrentInput(FrontendInput _currentIntput) {
  currentInput = _currentIntput;
}

std::string FrontendAction::getInputFile() {
  return std::string(currentInput.getInputFile());
}

std::string FrontendAction::getRemarksOutput() {
  return std::string(currentInput.getRemarksOutput());
}

llvm::Error FrontendAction::execute() {

  using namespace rust_compiler::session;

//  std::mt19937 gen32;
//  basic::CrateNum crateNum = gen32();

  basic::CrateNum crateNum = 1; // runParse
  Session currentSession = Session{crateNum, nullptr};
  rust_compiler::session::session = &currentSession;

  llvm::errs() << crateNum << "\n";

  tyctx::TyCtx ctx;
  ctx.setCurrentCrate(crateNum);

  rust_compiler::session::session->setTypeContext(&ctx);

  executeAction();

  return llvm::Error::success();
}

} // namespace rust_compiler::frontend

// FIXME: pcms
