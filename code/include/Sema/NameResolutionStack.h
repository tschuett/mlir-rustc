#pragma once

#include "ADT/Result.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Sema/Rib.h"

#include <map>
#include <optional>

namespace rust_compiler::sema {

enum class NameSpace { Values, Types, Labels, Macros };

template <NameSpace N> class NameResolutionStack {

public:
  NameResolutionStack()
      : root(Node(Rib(RibKind::Normal))), cursorReference(root){};

  void push(Rib rib, basic::NodeId id,
            std::optional<lexer::Identifier> path = {});

  void pop();

  adt::StringResult<basic::NodeId> insert(lexer::Identifier name,
                                          basic::NodeId id);

  Rib &peek();
  const Rib &peek() const;

  std::optional<basic::NodeId> get(const lexer::Identifier &name);

private:
  class Link {
  public:
    Link(basic::NodeId id, std::optional<lexer::Identifier> path)
        : id(id), path(path) {}

    bool compare(const Link &other) const { return id < other.id; }

    basic::NodeId id;
    std::optional<lexer::Identifier> path;
  };

  /* Link comparison class, which we use in a Node's `children` map */
  class LinkCmp {
  public:
    bool operator()(const Link &lhs, const Link &rhs) const {
      return lhs.compare(rhs);
    }
  };

  class Node {
  public:
    Node(Rib rib) : rib(rib) {}
    Node(Rib rib, Node &parent) : rib(rib), parent(parent) {}

    bool isRoot() const;
    bool isLeaf() const;

    void insertChild(Link link, Node child);

    Rib rib; // this is the "value" of the node - the data it keeps.
    std::map<Link, Node, LinkCmp> children; // all the other nodes it links to

    std::optional<Node &> parent; // `None` only if the node is a root
  };

  Node root;
  std::reference_wrapper<Node> cursorReference;
};

} // namespace rust_compiler::sema
