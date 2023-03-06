#pragma once

#include "Frontend/CompilerInvocation.h"

#include <clang/Basic/Diagnostic.h>
#include <memory>

namespace rust_compiler::frontend {

class CompilerInstance {
  std::unique_ptr<CompilerInvocation> invocation;

  /// The diagnostics engine instance.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics;

public:
  explicit CompilerInstance();

  CompilerInvocation &getInvocation() {
    assert(invocation && "Compiler instance has no invocation!");
    return *invocation;
  };

  /// Get the current diagnostics engine.
  clang::DiagnosticsEngine &getDiagnostics() const {
    assert(diagnostics && "Compiler instance has no diagnostics!");
    return *diagnostics;
  }

  /// Create the default output file (based on the invocation's options) and
  /// add it to the list of tracked output files. If the name of the output
  /// file is not provided, it will be derived from the input file.
  ///
  /// \param binary     The mode to open the file in.
  /// \param baseInput  If the invocation contains no output file name (i.e.
  ///                   outputFile in FrontendOptions is empty), the input path
  ///                   name to use for deriving the output path.
  /// \param extension  The extension to use for output names derived from
  ///                   \p baseInput.
  /// \return           Null on error, ostream for the output file otherwise
  std::unique_ptr<llvm::raw_pwrite_stream>
  createDefaultOutputFile(bool binary = true, llvm::StringRef baseInput = "",
                          llvm::StringRef extension = "");

private:
  /// Create a new output file
  ///
  /// \param outputPath   The path to the output file.
  /// \param binary       The mode to open the file in.
  /// \return             Null on error, ostream for the output file otherwise
  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>>
  createOutputFileImpl(llvm::StringRef outputPath, bool binary);
};

} // namespace rust_compiler::frontend
