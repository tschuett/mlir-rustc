#pragma once

#include "AST/Crate.h"
#include "Basic/Edition.h"
#include "Frontend/CompilerInstance.h"
#include "Frontend/FrontendOptions.h"

#include <llvm/Support/Error.h>
#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::frontend {

class FrontendAction {
  FrontendInput currentInput;
  CompilerInstance *instance;
  basic::Edition edition;
  std::shared_ptr<ast::Crate> crate;

protected:
  /// @name Implementation Action Interface
  /// @{

  /// Callback to run the program action, using the initialized
  /// compiler instance.
  virtual void executeAction() = 0;

  /// Callback at the start of processing a single input.
  ///
  /// \return True on success; on failure ExecutionAction() and
  /// EndSourceFileAction() will not be called.
  virtual bool beginSourceFileAction() { return true; }

  /// @}

public:
  CompilerInstance &getInstance() const {
    assert(instance && "Compiler instance not registered!");
    return *instance;
  }

  void setInstance(CompilerInstance *value) { instance = value; }

  virtual ~FrontendAction() = default;

  void setCurrentInput(FrontendInput currentIntput);

  void setEdition(basic::Edition edition);

  /// Run the action.
  llvm::Error execute();

  std::string getInputFile();

  std::string getRemarksOutput();

  ast::Crate *getCrate();

protected:
  // Parse the current input file. Return False if fatal errors are reported,
  // True otherwise.
  bool runParse();
  // Run semantic checks for the current input file. Return False if fatal
  // errors are reported, True otherwise.
  bool runSemanticChecks();
};

} // namespace rust_compiler::frontend
