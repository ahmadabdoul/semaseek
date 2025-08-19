# SemaSeek - Semantic Code Search

**SemaSeek** is a Visual Studio Code extension that provides natural-language semantic search for your codebase. Instead of relying on keywords, SemaSeek understands the *meaning* behind your search queries to find the most relevant code snippets.

This is a starter project that demonstrates how to build a semantic search extension using Google's Gemini embeddings.

## Features

*   **Semantic Search**: Use natural language to search your code. For example, you can search for "function to calculate fibonacci" instead of just "fibonacci".
*   **Automatic Indexing**: SemaSeek automatically indexes your workspace files in the background.
*   **File Watching**: The index is kept up-to-date by watching for file creations, changes, and deletions.
*   **Persistent Storage**: The search index can be persisted to a SQLite database, so it doesn't need to be rebuilt every time you open VS Code. A JSON file fallback is also available.
*   **Configurable**: You can configure the embedding model, batch size for API requests, and storage options.
*   **Secure API Key Storage**: Your Google API key is stored securely using VS Code's `SecretStorage`.

## Commands

*   `Semantic Search: Index Workspace`: Indexes the entire workspace.
*   `Semantic Search: Query`: Initiates a semantic search query.
*   `Semantic Search: Clear Index`: Deletes the search index.
*   `Semantic Search: Show Index Stats`: Shows the number of indexed chunks.
*   `Semantic Search: Set API Key`: Prompts for your Google API key and stores it securely.

## Configuration

You can configure SemaSeek through the VS Code settings (`settings.json`):

*   `semanticSearch.persistToSqlite` (default: `true`): Whether to persist the index to a SQLite database. If `false`, a JSON file is used as a fallback.
*   `semanticSearch.embeddingModel` (default: `"gemini-embedding-001"`): The embedding model to use. If you change this, you will need to re-index your workspace.
*   `semanticSearch.batchSize` (default: `20`): The batch size for embedding requests. You can adjust this to manage API rate limits and costs.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd semaseek
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Set your Google API Key**:
    *   Open the command palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
    *   Run the `Semantic Search: Set API Key` command.
    *   Paste your Google API key for the Gemini API.

4.  **Run the extension**:
    *   Press `F5` to open a new VS Code window with the extension running.

5.  **Index your workspace**:
    *   Open a workspace you want to index.
    *   Open the command palette and run `Semantic Search: Index Workspace`.

## Key Server (For Advanced Users)

The extension is configured to fetch a temporary API key from a key server, defined by the `KEY_SERVER_URL` constant in `src/extension.ts`. This is an advanced feature for managing API keys in a team environment.

If you are a solo developer, you can ignore this and use the `Semantic Search: Set API Key` command, which will always take precedence. If you want to set up your own key server, you will need to change the `KEY_SERVER_URL` to point to your server's endpoint.
