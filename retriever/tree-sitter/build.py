from tree_sitter import Language

Language.build_library(
  'build/my-languages.so',

  [
    'vendor/tree-sitter-java',
    'vendor/tree-sitter-python',
    'vendor/tree-sitter-cpp',
    'vendor/tree-sitter-c',
    'vendor/tree-sitter-go',
  ]
)