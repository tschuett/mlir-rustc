#pragma once

#include "AST/AttrInput.h"
#include "AST/SimplePath.h"

#include <memory>
#include <mutex>
#include <optional>

namespace rust_compiler::ast {

class Attr : public Node {
  SimplePath path;
  std::optional<AttrInput> attrInput;
  bool parsedToMetaItem = false;

public:
  Attr(Location loc) : Node(loc), path(loc) {}

  void setSimplePath(const SimplePath &sim) { path = sim; }
  void setAttrInput(const AttrInput &input) { attrInput = input; }

  void parseMetaItem();

  bool hasInput() const { return attrInput.has_value(); }
  AttrInput getInput() const { return *attrInput; }

  SimplePath getPath() const { return path; }

private:
  bool isParsedToMetaItem();
};

} // namespace rust_compiler::ast
