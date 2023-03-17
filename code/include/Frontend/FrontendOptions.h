#pragma once

#include <string>
#include <string_view>

namespace rust_compiler::frontend {

enum class ActionKind {
  /// --fsyntax-only
  ParseSyntaxOnly,

  /// --fwith-sema
  WithSema,

  /// Emit a .o file.
  EmitObj,
};

enum class InputKind { File, CargoTomlDir };

class FrontendInput {
  std::string inputFile;
  std::string remarksOutput;
  std::string crateName;
  InputKind kind;

public:
  FrontendInput() = default;
  FrontendInput(std::string_view inputFile, std::string_view remarksOutput,
                std::string_view crateName, InputKind inKind)
      : inputFile(inputFile), remarksOutput(remarksOutput),
        crateName(crateName), kind(inKind){};

  InputKind getKind() const { return kind; }

  std::string_view getInputFile() const { return inputFile; }
  std::string_view getRemarksOutput() const { return remarksOutput; }
  std::string_view getCrateName() const { return crateName; }
};

} // namespace rust_compiler::frontend
