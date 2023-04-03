#include "AST/MetaItemInner.h"

#include "AST/AttributeParser.h"
#include "Location.h"

namespace rust_compiler::ast {
std::unique_ptr<MetaNameValueString>
MetaItemInner::tryMetaNameValueString() const {
  if (isKeyValuePair())
    return static_cast<const MetaNameValueString *>(this)
        ->tryMetaNameValueString();

  // todo actually parse foo = bar
  return nullptr;
}

SimplePath MetaItemInner::tryPathItem() const {
  SimplePath path{Location::getEmptyLocation()};
  return path;
}

} // namespace rust_compiler::ast
