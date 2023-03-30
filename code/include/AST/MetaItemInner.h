#pragma once

namespace rust_compiler::ast {

class MetaItemInner {
public:
  virtual ~MetaItemInner() = default;

  virtual MetaItemInner *clone() = 0;
};

} // namespace rust_compiler::ast
