#pragma once

#include "AST/AST.h"
#include "AST/Abi.h"

#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

enum class FunctionQualifierKind { Const, Async, Unsafe, Extern };

class FunctionQualifiers : public Node {
  FunctionQualifierKind kind;
  std::optional<Abi> abi;

  bool isAsync = false;
  bool isConst = false;
  bool isUnsafe = false;
  bool isExtern = false;

public:
  FunctionQualifiers(Location loc) : Node(loc){};

  FunctionQualifierKind getKind() const { return kind; }

  void setAsync();
  void setConst();
  void setUnsafe();
  void setAbi(Abi _abi) { abi = _abi; }
  void setExtern() { isExtern = true; }
  bool hasExtern() const { return isExtern; }
};

} // namespace rust_compiler::ast
