#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/StructFields.h"

#include <optional>

namespace rust_compiler::ast {

class EnumItemStruct : public Node {
  std::optional<StructFields> fields;

public:
  EnumItemStruct(Location loc) : Node(loc) {}

  void setStructFields(const StructFields &sf) { fields = sf; }

  bool hasFields() const { return fields.has_value(); }
  StructFields getFields() const { return *fields; }
};

} // namespace rust_compiler::ast
