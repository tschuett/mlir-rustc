#pragma once

namespace rust_compiler::sema::type_checking {

class TypeCheckContext {
public:
  static TypeCheckContext *get();
};

} // namespace rust_compiler::sema::type_checking
