#pragma once

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
  std::string storage;

public:
  std::string toString() const;
  bool isASCII() const;
  bool isRawIdentifier() const;

  static Identifier fromString(std::string_view);

};

} // namespace rust_compiler::lexer
