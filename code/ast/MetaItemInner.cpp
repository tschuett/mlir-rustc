#include "AST/MetaItemInner.h"

#include "AST/AttributeParser.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {
std::optional<std::shared_ptr<MetaNameValueString>>
MetaItemInner::tryMetaNameValueString() const {
  if (isKeyValuePair())
    return static_cast<const MetaNameValueString *>(this)
        ->tryMetaNameValueString();

  // todo actually parse foo = bar
  return std::nullopt;
}

std::optional<SimplePath> MetaItemInner::tryPathItem() const {
  return std::nullopt;
}

} // namespace rust_compiler::ast
