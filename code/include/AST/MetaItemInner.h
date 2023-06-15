#pragma once

#include "AST/SimplePath.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class MetaNameValueString;
class SimplePath;

class MetaItemInner {
public:
  virtual ~MetaItemInner() = default;

  virtual MetaItemInner *clone() = 0;

  virtual std::optional<std::shared_ptr<MetaNameValueString>>
  tryMetaNameValueString() const;

  virtual std::optional<SimplePath> tryPathItem() const;

  virtual bool isKeyValuePair() const = 0;

  virtual bool isMetaItem() const { return false; };
};

} // namespace rust_compiler::ast
