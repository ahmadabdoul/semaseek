# SemaSeek - Semantic Code Search

**SemaSeek** is a Visual Studio Code extension that provides natural-language semantic search for your codebase. Instead of relying on keywords, SemaSeek understands the *meaning* behind your search queries to find the most relevant code snippets.

## Features

*   **Semantic Search**: Use natural language to search your code. For example, you can search for "function to calculate fibonacci" instead of just "fibonacci".
*   **Automatic Indexing**: SemaSeek automatically indexes your workspace files in the background after manually indexing on the first run (codebase).
*   **File Watching**: The index is kept up-to-date by watching for file creations, changes, and deletions.
*   **Persistent Storage**: The search index can be persisted to a SQLite database, so it doesn't need to be rebuilt every time you open VS Code. A JSON file fallback is also available.

## Commands

*   `Semantic Search: Index Workspace`: Indexes the entire workspace.
*   `Semantic Search: Query`: Initiates a semantic search query.
*   `Semantic Search: Clear Index`: Deletes the search index.
*   `Semantic Search: Show Index Stats`: Shows the number of indexed chunks.

## Configuration

You can configure SemaSeek through the VS Code settings (`settings.json`):

*   `semanticSearch.persistToSqlite` (default: `true`): Whether to persist the index to a SQLite database. If `false`, a JSON file is used as a fallback.
*   `semanticSearch.embeddingModel` (default: `"gemini-embedding-001"`): The embedding model to use. If you change this, you will need to re-index your workspace.
*   `semanticSearch.batchSize` (default: `20`): The batch size for embedding requests. You can adjust this to manage API rate limits and costs.

## Setup

1.  **Install Semaseek from the marketplace**
    


2.  **Index your workspace**:
    *   Open a workspace you want to index.
    *   Open the command palette and run `Semantic Search: Index Workspace`.


3.  **Query your codebase**:
    *   Open the command palette and run `Semantic Search: Query`.
    *   Type the search terms, phrase or keyword in the search bar
