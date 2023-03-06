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
  std::string file;
  std::string crateName;
  InputKind kind;

public:
  FrontendInput(std::string_view file, std::string_view crateName,
                InputKind inKind)
      : file(file), crateName(crateName), kind(inKind){};

  InputKind getKind() const { return kind; }

  std::string_view getFile() const { return file; }
  std::string_view getCrateName() const { return crateName; }
};

} // namespace rust_compiler::frontend
