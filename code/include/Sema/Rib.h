#pragma once

#include "ADT/Result.h"
#include "Basic/Ids.h"

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rust_compiler::sema {

enum class RibKind {
  Normal,
  Module,
  Function,
  ConstantItem,
  TraitOrImpl,
  Item,
  Closue,
  MacroDefinition,
  ConstParamType
};

class Rib {
  RibKind kind;

public:
  Rib(RibKind kind) : kind(kind) {}

  adt::StringResult<basic::NodeId>
  insert(std::string_view name, basic::NodeId id, bool canShadow = false);

  std::optional<basic::NodeId> get(std::string_view name);

  const std::unordered_map<std::string, basic::NodeId> &getValues() const {
    return values;
  }

private:
  std::unordered_map<std::string, basic::NodeId> values;
};

} // namespace rust_compiler::sema
