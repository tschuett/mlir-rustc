#pragma once

#include "AST/SimplePath.h"

#include <memory>

namespace rust_compiler::ast {

class MetaNameValueString;
class SimplePath;

class MetaItemInner {
public:
  virtual ~MetaItemInner() = default;

  virtual MetaItemInner *clone() = 0;

  std::unique_ptr<MetaNameValueString> tryMetaNameValueString() const;

  virtual SimplePath tryPathItem() const;
};

} // namespace rust_compiler::ast
