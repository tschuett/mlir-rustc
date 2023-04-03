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

  virtual std::unique_ptr<MetaNameValueString> tryMetaNameValueString() const;

  virtual SimplePath tryPathItem() const;

  virtual bool isKeyValuePair() const = 0;
};

} // namespace rust_compiler::ast
