/*
  semantic-code-search - starter VS Code extension (single-file starter)

  What's included (high-level):
  - workspace indexing command (walks files, chunks, creates embeddings via pluggable provider)
  - in-memory vector store with cosine-similarity search (placeholder - replace with sqlite/FAISS/etc.)
  - query command (input box -> semantic search -> quickpick results -> open editor at match)
  - simple stats & clear-index commands

  Notes:
  - Implement `getEmbedding(text)` to call your chosen embedding provider (OpenAI, local LLM, etc.).
  - For production, swap the in-memory store with a persisted vector DB (sqlite + vectordb, milvus, qdrant, or cloud-hosted).
  - Add package.json with activation events and commands. See comments at bottom for a minimal `package.json` snippet.
*/

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';

// ---------------------------
// Simple in-memory vector store
// ---------------------------

type DocChunk = {
  id: string; // unique id
  uri: string; // file uri
  start: number; // char index start
  end: number; // char index end
  text: string;
  embedding: number[] | null;
};

class InMemoryVectorStore {
  private items: DocChunk[] = [];

  add(chunk: DocChunk) {
    this.items.push(chunk);
  }

  clear() {
    this.items = [];
  }

  size() {
    return this.items.length;
  }

  async searchByEmbedding(embedding: number[], topK = 10) {
    const scored = this.items
      .filter(i => i.embedding && i.embedding.length === embedding.length)
      .map(i => ({ item: i, score: cosineSimilarity(i.embedding!, embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored;
  }
}

const vectorStore = new InMemoryVectorStore();

// ---------------------------
// Utilities
// ---------------------------

function cosineSimilarity(a: number[], b: number[]) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function chunkText(text: string, maxChars = 1200, overlap = 200) {
  const chunks: { start: number; end: number; text: string }[] = [];
  let pos = 0;
  while (pos < text.length) {
    const end = Math.min(text.length, pos + maxChars);
    const chunk = text.slice(pos, end);
    chunks.push({ start: pos, end, text: chunk });
    if (end === text.length) break;
    pos = Math.max(0, end - overlap);
  }
  return chunks;
}

// ---------------------------
// Embedding provider placeholder
// ---------------------------

// TODO: replace this function with your embedding provider implementation.
// Example options:
// - OpenAI embeddings (client SDK)
// - Local LLM embedding server
// - Lightweight on-device embedding

async function getEmbedding(text: string): Promise<number[]> {
  // Placeholder: returns a deterministic pseudo-embedding based on char codes.
  // THIS IS NOT SEMANTIC — replace it.
  const seed = 31;
  const hashLen = 128;
  const vec = new Array(hashLen).fill(0);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    vec[(i * seed) % hashLen] = (vec[(i * seed) % hashLen] + code) % 1000;
  }
  // normalize
  const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return vec.map(v => v / (mag || 1));
}

// ---------------------------
// Indexing logic
// ---------------------------

async function indexWorkspace(progress: vscode.Progress<{ message?: string; increment?: number }>, token: vscode.CancellationToken) {
  vectorStore.clear();

  // Adjust glob as desired
  const fileGlob = '**/*.{js,ts,jsx,tsx,py,java,go,rs,md}';
  const uris = await vscode.workspace.findFiles(fileGlob, '**/node_modules/**');

  const total = uris.length;
  let processed = 0;

  for (const uri of uris) {
    if (token.isCancellationRequested) break;
    try {
      const bytes = await vscode.workspace.fs.readFile(uri);
      const text = Buffer.from(bytes).toString('utf8');
      const chunks = chunkText(text);
      for (const c of chunks) {
        const id = `${uri.toString()}::${c.start}-${c.end}`;
        const chunk: DocChunk = { id, uri: uri.toString(), start: c.start, end: c.end, text: c.text, embedding: null };
        vectorStore.add(chunk);
      }
    } catch (e) {
      // ignore file read errors
    }
    processed++;
    const pct = Math.floor((processed / total) * 100);
    progress.report({ message: `Indexing ${uri.fsPath} (${processed}/${total})`, increment: pct });
  }

  // Batch-create embeddings (naive sequential implementation)
  const items = (vectorStore as any).items as DocChunk[];
  for (let i = 0; i < items.length; i++) {
    if (token.isCancellationRequested) break;
    try {
      const emb = await getEmbedding(items[i].text);
      items[i].embedding = emb;
    } catch (e) {
      items[i].embedding = null;
    }
    // optional: report progress every N items
    if (i % 50 === 0) {
      progress.report({ message: `Embedding chunks ${i}/${items.length}` });
    }
  }
}

// ---------------------------
// Commands
// ---------------------------

export async function activate(context: vscode.ExtensionContext) {
  console.log('semantic-code-search: activated');

  const indexCommand = vscode.commands.registerCommand('semanticSearch.indexWorkspace', async () => {
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Indexing workspace for semantic search', cancellable: true }, async (p, token) => {
      await indexWorkspace(p, token);
      vscode.window.showInformationMessage(`Indexed ${vectorStore.size()} text chunks for semantic search.`);
    });
  });

  const queryCommand = vscode.commands.registerCommand('semanticSearch.query', async () => {
    if (vectorStore.size() === 0) {
      const pick = await vscode.window.showInformationMessage('Index empty. Index workspace now?', 'Index', 'Cancel');
      if (pick === 'Index') {
        await vscode.commands.executeCommand('semanticSearch.indexWorkspace');
      } else return;
    }

    const q = await vscode.window.showInputBox({ prompt: 'Search code semantically (natural language)' });
    if (!q) return;

    const qEmb = await getEmbedding(q);
    const results = await vectorStore.searchByEmbedding(qEmb, 12);

    if (results.length === 0) {
      vscode.window.showInformationMessage('No matches found.');
      return;
    }

    const items = results.map(r => ({
      label: `${(r.score * 100).toFixed(1)}% — ${path.basename(r.item.uri)}`,
      description: truncate(r.item.text.replace(/\s+/g, ' '), 200),
      uri: r.item.uri,
      start: r.item.start,
      score: r.score
    }));

    const pick = await vscode.window.showQuickPick(items, { placeHolder: 'Select a match to open' });
    if (!pick) return;

    const doc = await vscode.workspace.openTextDocument(vscode.Uri.parse(pick.uri));
    const editor = await vscode.window.showTextDocument(doc);

    const pos = doc.positionAt(pick.start);
    editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
    editor.selection = new vscode.Selection(pos, pos);
  });

  const clearCommand = vscode.commands.registerCommand('semanticSearch.clearIndex', async () => {
    vectorStore.clear();
    vscode.window.showInformationMessage('Semantic index cleared.');
  });

  const statsCommand = vscode.commands.registerCommand('semanticSearch.showStats', async () => {
    vscode.window.showInformationMessage(`Indexed chunks: ${vectorStore.size()}`);
  });

  context.subscriptions.push(indexCommand, queryCommand, clearCommand, statsCommand);
}

export function deactivate() {
  console.log('semantic-code-search: deactivated');
}

// small helpers
function truncate(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

/*
  Minimal package.json snippet (add this to your extension's package.json):

  {
    "name": "semantic-code-search",
    "displayName": "Semantic Code Search (Starter)",
    "description": "Starter VS Code extension for semantic search in a workspace (pluggable embeddings).",
    "version": "0.0.1",
    "engines": { "vscode": "^1.70.0" },
    "activationEvents": [
      "onCommand:semanticSearch.indexWorkspace",
      "onCommand:semanticSearch.query"
    ],
    "main": "./out/extension.js",
    "contributes": {
      "commands": [
        { "command": "semanticSearch.indexWorkspace", "title": "Semantic Search: Index Workspace" },
        { "command": "semanticSearch.query", "title": "Semantic Search: Query" },
        { "command": "semanticSearch.clearIndex", "title": "Semantic Search: Clear Index" },
        { "command": "semanticSearch.showStats", "title": "Semantic Search: Show Index Stats" }
      ]
    },
    "scripts": {
      "vscode:prepublish": "npm run compile",
      "compile": "tsc -p ./",
      "watch": "tsc -watch -p ./"
    },
    "devDependencies": {
      "typescript": "^4.0.0",
      "@types/vscode": "^1.70.0",
      "vscode-test": "^1.6.0"
    }
  }

  Remember to create a tsconfig.json for the extension build and wire up the build to produce ./out/extension.js.
*/
