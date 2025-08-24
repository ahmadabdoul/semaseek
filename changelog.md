# Changelog

## [0.1.3] - 2025-08-24

### Added
- **Hybrid Search:** Implemented a hybrid search model that combines semantic search with keyword-based search for more accurate results.
- **Syntax-Aware Chunking:** The extension now uses `tree-sitter` to create intelligent, syntax-aware code chunks, significantly improving the quality of embeddings.
- **Configuration Options:** Added settings to enable/disable hybrid search and to configure the lexical boost weight.

### Changed
- **Database Engine:** Replaced the in-memory vector store and SQLite implementation with **LanceDB**, a high-performance vector database, to improve scalability and performance with large codebases.
