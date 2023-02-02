#pragma once

#include <string>
#include <vector>

namespace rust_compiler::adt {
  class ScopedCanonicalPath;
}

namespace rust_compiler::adt {

class CanonicalPath {
  std::vector<std::string> segments;
  std::string crateName;

public:
  CanonicalPath(std::string_view crateName) : crateName(crateName) {}

  CanonicalPath append(std::string_view segment) const {
    if (isEmpty()) {
      std::vector<std::string> segs;
      segs.push_back(std::string(segment));
      auto canPath = CanonicalPath(crateName);
      canPath.segments = segs;
      return canPath;
    }

    std::vector<std::string> copy(segments);
    copy.push_back(std::string(segment));

    auto canPath = CanonicalPath(crateName);
    canPath.segments = copy;
    return canPath;
  }

  std::string asString() const {
    std::string buf;
    for (size_t i = 0; i < segments.size(); i++) {
      bool have_more = (i + 1) < segments.size();
      const std::string &seg = segments.at(i);
      buf += seg + (have_more ? "::" : "");
    }
    return buf;
  }

  bool isEmpty() const { return segments.size() == 0; }

private:
  friend ScopedCanonicalPath;
};

} // namespace rust_compiler::adt
