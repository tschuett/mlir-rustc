#pragma once

#include "Basic/Edition.h"
#include "Frontend/CompilerInstance.h"
#include "Frontend/FrontendOptions.h"

#include <llvm/Support/Error.h>
#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast {
class Crate;
}
namespace rust_compiler::frontend {

class FrontendAction {
  FrontendInput currentInput;
  CompilerInstance *instance;

  std::shared_ptr<ast::Crate> crate;

protected:
  /// @name Implementation Action Interface
  /// @{

  /// Callback to run the program action, using the initialized
  /// compiler instance.
  virtual void executeAction() = 0;

  /// Callback at the end of processing a single input, to determine
  /// if the output files should be erased or not.
  ///
  /// By default it returns true if a compiler error occurred.
  virtual bool shouldEraseOutputFiles();

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

  virtual ~FrontendAction() = default;

  void setCurrentInput(FrontendInput currentIntput);

  void setEdition(basic::Edition edition);

  /// Run the action.
  llvm::Error execute();

protected:
  // Parse the current input file. Return False if fatal errors are reported,
  // True otherwise.
  bool runParse();
  // Run semantic checks for the current input file. Return False if fatal
  // errors are reported, True otherwise.
  bool runSemanticChecks();
};

} // namespace rust_compiler::frontend
