#pragma once

#include "Frontend/CompilerInstance.h"

namespace rust_compiler::frontend {

class FrontendAction {
  CompilerInstance *instance;

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

  void setInstance(CompilerInstance *value) { instance = value; }
};

} // namespace rust_compiler::frontend
