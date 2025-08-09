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
let genaiClient: any | null = null;

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

/**
 * Initialize GenAI client (one-time).
 * Priority order for API key:
 *  1) context.secrets (recommended for installed extension)
 *  2) process.env.GOOGLE_API_KEY (dev or CI)
 */
async function initGenAI(context?: vscode.ExtensionContext) {
  if (genaiClient) return genaiClient;

  // try SecretStorage (recommended)
  let apiKey: string | undefined;
  try {
    if (context?.secrets) {
      apiKey = await context.secrets.get('GOOGLE_API_KEY') || undefined;
    }
  } catch {
    apiKey = undefined;
  }

  // fallback to env var (local dev)
  if (!apiKey && process.env.GOOGLE_API_KEY) {
    apiKey = process.env.GOOGLE_API_KEY;
  }

  if (!apiKey) {
    // Throwing lets callers surface a helpful message to user
    throw new Error(
      'Google API key not found. Store one in VS Code SecretStorage under key GOOGLE_API_KEY, or set env GOOGLE_API_KEY.'
    );
  }

  // lazy import SDK so installing the extension without the package doesn't break activation
  const { Models, Client } = await import('@google/genai'); // package exports
  // Most examples use a single `ai` client with .models.* methods.
  // create client: SDK will accept apiKey param
  // NOTE: exact constructor shape can vary by version; this follows official docs.
  const ai = new (Client ?? (Models && Models))({ apiKey }); // defensive fallback
  genaiClient = ai;
  return genaiClient;
}

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


export async function getEmbedding(text: string | string[], context?: vscode.ExtensionContext): Promise<number[] | number[][]> {
    const client = await initGenAI(context);
  
    // Normalize to array of strings
    const contents = Array.isArray(text) ? text : [text];
  
    // Choose model (gemini-embedding-001 recommended). You can make this configurable.
    const MODEL = 'gemini-embedding-001';
  
    // Call embedContent API (batch)
    let resp: any;
    try {
      resp = await client.models.embedContent({
        model: MODEL,
        contents,            // array of inputs
        // optional config:
        // config: { outputDimensionality: 1536 } // to reduce to 1536 dims, if you want
      });
    } catch (err) {
      // surface useful error
      const message = (err && err.message) ? err.message : String(err);
      throw new Error(`Gemini embedContent failed: ${message}`);
    }
  
    // Defensive: extract embeddings for each input
    // Common shapes (depending on SDK version):
    //  - resp.embeddings -> array of arrays
    //  - resp.data[0].embedding or resp.data[i].embedding (object) -> might contain .values
    const tryExtract = (r: any): number[][] | null => {
      if (!r) return null;
      if (Array.isArray(r.embeddings) && r.embeddings.length > 0 && Array.isArray(r.embeddings[0])) {
        return r.embeddings as number[][];
      }
      if (Array.isArray(r.data) && r.data.length > 0) {
        // common pattern: data[i].embedding or data[i].embedding.values
        const out: number[][] = [];
        for (const d of r.data) {
          if (!d) continue;
          if (Array.isArray(d.embedding)) out.push(d.embedding as number[]);
          else if (Array.isArray(d.embedding?.values)) out.push(d.embedding.values as number[]);
          else if (Array.isArray(d.embedding?.embedding)) out.push(d.embedding.embedding as number[]);
          else if (Array.isArray(d.values)) out.push(d.values as number[]);
          else if (Array.isArray(d.embedding?.vector)) out.push(d.embedding.vector as number[]);
        }
        if (out.length > 0) return out;
      }
      // some SDKs return an outputs array
      if (Array.isArray(r.outputs) && r.outputs.length > 0 && Array.isArray(r.outputs[0].embedding)) {
        return r.outputs.map((o: any) => o.embedding as number[]);
      }
      return null;
    };
  
    const extracted = tryExtract(resp);
    if (!extracted) {
      // last attempt: if response itself is a flat numerical array (single input)
      if (Array.isArray(resp) && typeof resp[0] === 'number') {
        return [resp as number[]];
      }
      // couldn't parse
      throw new Error('Unexpected Gemini embeddings response shape — check SDK docs or log full response.');
    }
  
    // If caller asked for single input, return single vector
    if (!Array.isArray(text)) return extracted[0];
    return extracted;
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
