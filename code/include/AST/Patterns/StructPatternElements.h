#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast::patterns {

class StructPatternEtCetera : public Node {
  std::vector<OuterAttribute> outerAttributes;

public:
  StructPatternEtCetera(Location loc) : Node(loc) {}
};

class StructPatternField : public Node {
  std::vector<OuterAttribute> outerAttributes;

public:
  StructPatternField(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }
};

class StructPatternFields : public Node {
  std::vector<StructPatternField> fields;

public:
  StructPatternFields(Location loc) : Node(loc) {}

  void addPattern(const StructPatternField &f) { fields.push_back(f); }
};

class StructPatternElements : public Node {

public:
  StructPatternElements(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast::patterns
