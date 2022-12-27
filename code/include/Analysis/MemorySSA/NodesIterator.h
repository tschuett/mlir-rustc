#pragma once

#include "Analysis/MemorySSA/Node.h"
//#include "Analysis/MemorySSA/NodeType.h"

#include <llvm/ADT/simple_ilist.h>

namespace rust_compiler::analysis {

class NodesIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Node;
  using pointer = value_type *;
  using reference = value_type &;

  using internal_iterator = llvm::simple_ilist<Node>::iterator;

  NodesIterator(internal_iterator iter);
  NodesIterator(const NodesIterator &) = default;
  NodesIterator(NodesIterator &&) = default;

  NodesIterator &operator=(const NodesIterator &) = default;
  NodesIterator &operator=(NodesIterator &&) = default;

  bool operator==(const NodesIterator &rhs) const {
    return iterator == rhs.iterator;
  }
  bool operator!=(const NodesIterator &rhs) const {
    return iterator != rhs.iterator;
  }

  NodesIterator &operator++();
  NodesIterator operator++(int);

  NodesIterator &operator--();
  NodesIterator operator--(int);

  reference operator*();
  pointer operator->();

private:
  internal_iterator iterator;
};

} // namespace rust_compiler::analysis
