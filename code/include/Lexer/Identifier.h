#pragma once

#include "ADT/Utf8String.h"

#include <string>
#include <string_view>

namespace rust_compiler::lexer {

/// https://doc.rust-lang.org/reference/identifiers.html
///
/// * foo
/// * _identifier
/// * r#true
/// * Москва
/// * 東京
class Identifier {
  adt::Utf8String storage;

public:
  Identifier() = default;

  /// must be ASCII. No UTF-8
  Identifier(std::string_view);

  /// utf8 converter to ascii
  std::string toString() const;
  bool isASCII() const;
  bool isRawIdentifier() const;

  bool operator==(const Identifier &b) const { return storage == b.storage; }

  bool operator<(const Identifier &b) const {
    return storage < b.storage;
  }

private:
};

} // namespace rust_compiler::lexer
