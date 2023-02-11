#pragma once

#include "Basic/Ids.h"

#include <string>
#include <vector>

namespace rust_compiler::adt {
class ScopedCanonicalPath;
}

/// https://doc.rust-lang.org/reference/paths.html#canonical-paths

namespace rust_compiler::adt {

class CanonicalPath {
  std::vector<std::pair<basic::NodeId, std::string>> segments;
  basic::CrateNum crateNum;

public:
  static CanonicalPath newSegment(basic::NodeId id, std::string_view path) {
    assert(!path.empty());
    return CanonicalPath({std::pair<basic::NodeId, std::string>(id, path)},
                         basic::UNKNOWN_CREATENUM);
  }

  std::string asString() const {
    std::string buf;
    for (size_t i = 0; i < segments.size(); i++) {
      bool haveMore = (i + 1) < segments.size();
      const std::string &seg = segments.at(i).second;
      buf += seg + (haveMore ? "::" : "");
    }
    return buf;
  }

  CanonicalPath append(const CanonicalPath &other) const {
    assert(!other.isEmpty());
    if (isEmpty())
      return CanonicalPath(other.segments, crateNum);

    std::vector<std::pair<basic::NodeId, std::string>> copy(segments);
    for (auto &s : other.segments)
      copy.push_back(s);

    return CanonicalPath(copy, crateNum);
  }

  bool isEmpty() const { return segments.size() == 0; }

private:
  explicit CanonicalPath(
      std::vector<std::pair<basic::NodeId, std::string>> path,
      basic::CrateNum crateNum)
      : segments(path), crateNum(crateNum) {}

  friend ScopedCanonicalPath;
};

} // namespace rust_compiler::adt
