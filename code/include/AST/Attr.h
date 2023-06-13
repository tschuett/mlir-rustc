#pragma once

#include "AST/AttrInput.h"
#include "AST/SimplePath.h"

#include <memory>
#include <mutex>
#include <optional>

namespace rust_compiler::ast {

class Attr : public Node {
  SimplePath path;
  std::unique_ptr<AttrInput> attrInput;

public:
  Attr(Location loc) : Node(loc), path(loc) {}

  void setSimplePath(const SimplePath &sim) { path = sim; }
  void setAttrInput(std::unique_ptr<AttrInput> input) {
    attrInput = std::move(input);
  }

  void parseMetaItem();

  Attr(const Attr &other);

  Attr &operator=(const Attr &other);

  // default move semantics
  Attr(Attr &&other) = default;
  Attr &operator=(Attr &&other) = default;

  AttrInput *getInput() const { return attrInput.get(); }

  SimplePath getPath() const { return path; }
private:
  bool isParsedToMetaItem();
  bool hasAttrInput();
};

} // namespace rust_compiler::ast
