#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/Patterns/Pattern.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast::patterns {

class StructPatternEtCetera : public Node {
  std::vector<OuterAttribute> outerAttributes;

public:
  StructPatternEtCetera(Location loc) : Node(loc) {}
};

enum class StructPatternFieldKind { TupleIndex, Identifier, RefMut };

class StructPatternField : public Node {
  std::vector<OuterAttribute> outerAttributes;
  StructPatternFieldKind kind;
  std::string tupleIndex;
  std::string identifier;
  bool ref = false;
  bool mut = false;
  std::shared_ptr<ast::patterns::Pattern> pattern;

public:
  StructPatternField(Location loc) : Node(loc) {}

  void setKind(StructPatternFieldKind kin) { kind = kin; }
  void setTupleIndex(std::string_view idx) { tupleIndex = idx; }
  void setIdentifier(std::string_view i) { identifier = i; }
  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  };

  void setPattern(std::shared_ptr<ast::patterns::Pattern> pat) {
    pattern = pat;
  }

  void setRef() { ref = true; }
  void setMut() { mut = true; }
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
