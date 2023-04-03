#pragma once

#include <span>
#include <vector>

namespace rust_compiler::target_info {

class Type;

class Signature {
  [[maybe_unused]]Type *returnType;
  std::vector<Type *> arguments;

public:
  Signature(Type *returnType, std::span<Type *> arguments2)
      : returnType(returnType) {
    arguments.assign(arguments2.begin(), arguments2.end());
  }
};

} // namespace rust_compiler::target_info
