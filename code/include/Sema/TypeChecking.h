#pragma once

#include <memory>

namespace rust_compiler::ast {
class Expression;
}

namespace rust_compiler::sema {

class TypeChecking {

public:


  void eq();
  void sub();
};

} // namespace rust_compiler::sema
