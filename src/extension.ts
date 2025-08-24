/**
  extension.ts - semantic-code-search (uses key server for API key)

  - Uses Gemini/embedContent via @google/genai
  - L2-normalizes embeddings for stable cosine search
  - Persists to async sqlite3 with JSON fallback
  - Key is fetched from a simple remote key server (no secret header used here)
  - File watcher + per-file status updates
*/

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as https from 'https';
import axios from 'axios';
import * as lancedb from '@lancedb/lancedb';
import { chunkCode } from './chunker';
import { SearchManager } from './search';

// Google GenAI SDK (typed import kept loose)
import { GoogleGenAI } from '@google/genai';

let genaiClient: GoogleGenAI | null = null;
let fileUpdateStatus: vscode.StatusBarItem | null = null;
const API_KEY = "";
// ---------------------------
// KEY SERVER CONFIG
// ---------------------------
// Replace this with your real server URL (no trailing slash recommended).
const KEY_SERVER_URL = 'http://pottersheritage.com/semaseek/semaseek.php'; // <-- ensure no trailing slash
const KEY_REFRESH_MARGIN_MS = 2 * 60 * 1000; // 2 minutes margin before expiry to refresh

// in-memory cached API key (don't persist per request)
let cachedApiKey: string | null = null;
let cachedApiKeyExpiresAt: number | null = null;

// ---------------------------
// VectorDB
// ---------------------------

type DocChunk = {
    id: string;
    uri: string;
    start: number;
    end: number;
    text: string;
    vector: number[] | null;
  };

  class VectorDB {
    private db!: lancedb.Connection;
    private table!: lancedb.Table;

    async init(context: vscode.ExtensionContext) {
      const dir = path.join(context.globalStorageUri.fsPath, 'lancedb');
      await fs.mkdir(dir, { recursive: true });
      this.db = await lancedb.connect(dir);
    }

    async getTable() {
      if (this.table) {
        return this.table;
      }
      const tables = await this.db.tableNames();
      if (!tables.includes('vectors')) {
        // The schema is inferred from the first batch of data
        this.table = await this.db.createTable('vectors', [{
            id: "1",
            uri: "1",
            start: 1,
            end: 1,
            text: "1",
            vector: Array(768).fill(1),
        }]);
        await this.table.delete("1=1")
      } else {
        this.table = await this.db.openTable('vectors');
      }

      return this.table;
    }
  }

  const vectorDB = new VectorDB();

// ---------------------------
// Utilities
// ---------------------------

const output = vscode.window.createOutputChannel('Semantic Search');

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  try { return String(err); } catch { return 'Unknown error'; }
}

function l2Normalize(v: number[]) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum) || 1;
  return v.map(x => x / norm);
}

function getLanguageId(uri: vscode.Uri): string {
    const extension = path.extname(uri.fsPath);
    switch (extension) {
        case '.ts':
        case '.tsx':
            return 'typescript';
        case '.js':
        case '.jsx':
            return 'javascript';
        case '.md':
            return 'markdown';
        // Add other languages as needed
        default:
            return 'plaintext';
    }
}

function normalizeEmbeddingResult(res: number[] | number[][]): number[] {
  if (Array.isArray(res) && res.length > 0 && Array.isArray(res[0])) {
    return (res as number[][])[0];
  }
  return res as number[];
}

// ---------------------------
// GenAI client + embedding (per docs) using key server
// ---------------------------

async function initGenAI(context?: vscode.ExtensionContext) {
    const ai = new GoogleGenAI({ apiKey: API_KEY });
    genaiClient = ai;
    return genaiClient;
}

/**
 * getEmbedding: uses gemini-embedding-001 (stable)
 * Returns normalized vectors (number[] or number[][])
 */
export async function getEmbedding(text: string | string[], context?: vscode.ExtensionContext): Promise<number[] | number[][]> {
  const client = await initGenAI(context);
  const contents = Array.isArray(text) ? text : [text];
  const MODEL = 'gemini-embedding-001';
  const TASK = 'SEMANTIC_SIMILARITY';

  try {
    const resp: any = await (client as any).models.embedContent({ model: MODEL, contents, taskType: TASK });

    // SDK shapes vary; extract defensively and normalize
    if (Array.isArray(resp?.embeddings) && resp.embeddings.length > 0) {
      const extracted = resp.embeddings.map((e: any) => {
        if (Array.isArray(e?.values)) return e.values as number[];
        if (Array.isArray(e?.embedding?.values)) return e.embedding.values as number[];
        if (Array.isArray(e?.embedding)) return e.embedding as number[];
        if (Array.isArray(e?.values)) return e.values as number[];
        if (Array.isArray(e)) return e as number[];
        throw new Error('Unexpected embedding entry shape');
      });
      const normalized = extracted.map((v: any[]) => l2Normalize(v.map((n: any) => Number(n))));
      return Array.isArray(text) ? normalized : normalized[0];
    }

    if (Array.isArray(resp?.data) && resp.data.length > 0) {
      const extracted2 = resp.data.map((d: any) => {
        if (Array.isArray(d?.embedding?.values)) return d.embedding.values as number[];
        if (Array.isArray(d?.embedding)) return d.embedding as number[];
        if (Array.isArray(d?.values)) return d.values as number[];
        if (Array.isArray(d)) return d as number[];
        throw new Error('Unexpected data.embedding shape');
      });
      const normalized2 = extracted2.map((v: any[]) => l2Normalize(v.map((n: any) => Number(n))));
      return Array.isArray(text) ? normalized2 : normalized2[0];
    }

    if (!Array.isArray(text) && Array.isArray(resp) && typeof resp[0] === 'number') return l2Normalize((resp as number[]).map(n => Number(n)));
    throw new Error('Unexpected embeddings response shape');
  } catch (err) {
    throw new Error(`Gemini embedContent failed: ${getErrorMessage(err)}`);
  }
}


// ---------------------------
// Update single file index (used by file watcher)
// ---------------------------

const pendingUpdates: Map<string, any> = new Map();
async function updateFileIndex(uri: vscode.Uri, searchManager: SearchManager, context?: vscode.ExtensionContext) {
  try {
    if (fileUpdateStatus) {
        fileUpdateStatus.text = 'Semantic Search: Updating ' + path.basename(uri.fsPath);
    }
    fileUpdateStatus?.show();

    output.appendLine(`Updating index for ${uri.fsPath}`);
    // read file
    const bytes = await vscode.workspace.fs.readFile(uri);
    const text = Buffer.from(bytes).toString('utf8');
    const chunks = chunkCode(text, getLanguageId(uri));
    const fileChunks: DocChunk[] = chunks.map(c => ({ id: `${uri.toString()}::${c.start}-${c.end}`, uri: uri.toString(), start: c.start, end: c.end, text: c.text, vector: null }));

    // fetch embeddings in batches
    const BATCH_SIZE = vscode.workspace.getConfiguration('semanticSearch').get<number>('batchSize', 20);
    for (let i = 0; i < fileChunks.length; i += BATCH_SIZE) {
      const batch = fileChunks.slice(i, i + BATCH_SIZE);
      try {
        const resp = await getEmbedding(batch.map(b => b.text), context);
        if (Array.isArray(resp) && Array.isArray(resp[0])) {
          for (let j = 0; j < batch.length; j++) batch[j].vector = (resp as number[][])[j] || null;
        } else if (Array.isArray(resp) && typeof resp[0] === 'number' && batch.length === 1) {
          batch[0].vector = resp as number[];
        }
      } catch (err) {
        output.appendLine('Failed to embed file batch: ' + getErrorMessage(err));
        // leave embeddings null for these chunks
      }
    }

    const table = await vectorDB.getTable();
    await table.delete(`uri = "${uri.toString()}"`);
    const validChunks = fileChunks.filter(c => c.vector);
    await table.add(validChunks);

    // Update keyword index
    validChunks.forEach(c => searchManager.add(c.id, c.text));


    output.appendLine(`Updated index for ${uri.fsPath} (${fileChunks.length} chunks)`);
  } catch (err) {
    output.appendLine(`Failed to update index for ${uri.fsPath}: ${getErrorMessage(err)}`);
  } finally {
    pendingUpdates.delete(uri.toString());
    // small delay to keep status visible for a moment
    setTimeout(() => fileUpdateStatus?.hide(), 400);
  }
}

function scheduleFileUpdate(uri: vscode.Uri, searchManager: SearchManager, context?: vscode.ExtensionContext) {
  const key = uri.toString();
  const debounceMs = 1000; // 1s debounce
  if (pendingUpdates.has(key)) {
    clearTimeout(pendingUpdates.get(key));
  }
  if (fileUpdateStatus) {
    fileUpdateStatus.text = 'Semantic Search: Queued update for ' + path.basename(uri.fsPath);
  }
  fileUpdateStatus?.show();
  const t = setTimeout(() => updateFileIndex(uri, searchManager, context), debounceMs);
  pendingUpdates.set(key, t);
}

// ---------------------------
// Indexing (workspace)
// ---------------------------

async function indexWorkspace(progress: vscode.Progress<{ message?: string; increment?: number }>, token: vscode.CancellationToken, searchManager: SearchManager, context?: vscode.ExtensionContext) {
  output.clear();
  output.appendLine('Starting workspace indexing...');
  searchManager.clear();

  const fileGlob = vscode.workspace.getConfiguration('semanticSearch').get<string>('fileGlob', '**/*.{js,ts,jsx,tsx,py,java,go,rs,md}');
  const uris = await vscode.workspace.findFiles(fileGlob, '**/node_modules/**');

  output.appendLine(`Found ${uris.length} files.`);
  const total = uris.length;
  let processed = 0;
  const allChunks: DocChunk[] = [];

  for (const uri of uris) {
    if (token.isCancellationRequested) break;
    try {
      if (fileUpdateStatus) {
        fileUpdateStatus.text = 'Semantic Search: Indexing ' + path.basename(uri.fsPath);
      }
      fileUpdateStatus?.show();

      const bytes = await vscode.workspace.fs.readFile(uri);
      const text = Buffer.from(bytes).toString('utf8');
      const chunks = chunkCode(text, getLanguageId(uri));
      for (const c of chunks) {
        const id = `${uri.toString()}::${c.start}-${c.end}`;
        const chunk: DocChunk = { id, uri: uri.toString(), start: c.start, end: c.end, text: c.text, vector: null };
        allChunks.push(chunk);
      }
    } catch (e) {
      output.appendLine(`Error reading ${uri.fsPath}: ${getErrorMessage(e)}`);
    }
    processed++;
    progress.report({ message: `Indexing ${uri.fsPath}`, increment: (processed / Math.max(1, total)) * 50 });
  }

  output.appendLine(`Created ${allChunks.length} chunks. Starting embedding...`);

  const items = allChunks;
  const BATCH_SIZE = vscode.workspace.getConfiguration('semanticSearch').get<number>('batchSize', 20);
  const maxAttempts = 6;
  const totalBatches = Math.max(1, Math.ceil(items.length / BATCH_SIZE));

  for (let i = 0, batchIdx = 0; i < items.length; i += BATCH_SIZE, batchIdx++) {
    if (token.isCancellationRequested) break;

    const batch = items.slice(i, i + BATCH_SIZE);
    const texts = batch.map(b => b.text);

    let embeddings: number[][] | null = null;
    let attempt = 0;

    while (attempt < maxAttempts) {
      try {
        attempt++;
        output.appendLine(`Embedding batch ${batchIdx + 1}/${totalBatches} (attempt ${attempt})`);
        const resp = await getEmbedding(texts, context);
        if (Array.isArray(resp) && Array.isArray(resp[0])) { embeddings = resp as number[][]; break; }
        throw new Error('Unexpected embedding shape');
      } catch (err) {
        const msg = getErrorMessage(err);
        output.appendLine(`Error embedding batch ${batchIdx + 1}: ${msg}`);
        if (attempt >= maxAttempts) { output.appendLine(`Giving up on batch ${batchIdx + 1}`); break; }
        const backoffMs = Math.pow(2, attempt) * 500 + Math.floor(Math.random() * 500);
        output.appendLine(`Retrying in ${backoffMs}ms...`);
        await new Promise(res => setTimeout(res, backoffMs));
      }
    }

    if (embeddings) {
      for (let j = 0; j < batch.length; j++) batch[j].vector = embeddings[j] || null;
    } else for (const b of batch) b.vector = null;

    const embeddingProgressIncrement = 50 / totalBatches;
    progress.report({ message: `Embedding chunks ${Math.min(i + BATCH_SIZE, items.length)}/${items.length}`, increment: embeddingProgressIncrement });
    await new Promise(res => setTimeout(res, 200));
  }

  const table = await vectorDB.getTable();
  const validChunks = items.filter(c => c.vector);
  await table.add(validChunks);
  validChunks.forEach(c => searchManager.add(c.id, c.text));

  // hide file update status when finished
  setTimeout(() => fileUpdateStatus?.hide(), 200);

  output.appendLine('Indexing complete.'); output.show(true);
}

// ---------------------------
// Commands & activation
// ---------------------------

export async function activate(context: vscode.ExtensionContext) {
  output.appendLine('Activating semantic-code-search extension');

  await vectorDB.init(context);
  const searchManager = new SearchManager();

  const indexCommand = vscode.commands.registerCommand('semanticSearch.indexWorkspace', async () => {
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Indexing workspace for semantic search', cancellable: true }, async (p, token) => {
      await indexWorkspace(p, token, searchManager, context);
      const table = await vectorDB.getTable();
      const count = await table.countRows();
      vscode.window.showInformationMessage(`Indexed ${count} text chunks for semantic search.`);
    });
  });

  const queryCommand = vscode.commands.registerCommand('semanticSearch.query', async () => {
    const table = await vectorDB.getTable();
    const count = await table.countRows();
    if (count === 0) {
        const pick = await vscode.window.showInformationMessage('Index empty. Index workspace now?', 'Index', 'Cancel');
        if (pick === 'Index') { await vscode.commands.executeCommand('semanticSearch.indexWorkspace'); } else return;
    }

    const q = await vscode.window.showInputBox({ prompt: 'Search code semantically (natural language)' });
    if (!q) return;

    let qEmbRaw: number[] | number[][];
    try { qEmbRaw = await getEmbedding(q, context); } catch (err) { vscode.window.showErrorMessage('Failed to get query embedding: ' + getErrorMessage(err)); return; }

    const qEmb = l2Normalize(normalizeEmbeddingResult(qEmbRaw));

    type SearchResult = {
      id: string;
      uri: string;
      start: number;
      end: number;
      text: string;
      _distance: number;
      score?: number;
    };

    const semanticResults = await table.search(qEmb).limit(20).toArray() as SearchResult[];

    const enableHybridSearch = vscode.workspace.getConfiguration('semanticSearch').get<boolean>('enableHybridSearch', true);
    let results: SearchResult[];

    if (enableHybridSearch) {
        const keywordResults = searchManager.search(q);
        const lexicalBoostWeight = vscode.workspace.getConfiguration('semanticSearch').get<number>('lexicalBoostWeight', 0.12);
        results = semanticResults
            .map(r => {
                const isKeywordMatch = keywordResults.includes(r.id);
                const score = 1 - r._distance;
                const finalScore = isKeywordMatch ? score + lexicalBoostWeight : score;
                return { ...r, score: finalScore };
            })
            .sort((a, b) => b.score! - a.score!)
            .slice(0, 12);
    } else {
        results = semanticResults
            .map(r => ({ ...r, score: 1 - r._distance }))
            .sort((a, b) => b.score! - a.score!)
            .slice(0, 12);
    }

    if (results.length === 0) { vscode.window.showInformationMessage('No matches found.'); return; }

    const items = results.map(r => ({ label: `${(r.score! * 100).toFixed(1)}% — ${path.basename(r.uri)}`, description: truncate(r.text.replace(/\s+/g, ' '), 200), uri: r.uri, start: r.start, end: r.end, score: r.score! }));

    const pick = await vscode.window.showQuickPick(items, { placeHolder: 'Select a match to open' });
    if (!pick) return;

    const doc = await vscode.workspace.openTextDocument(vscode.Uri.parse(pick.uri));
    const editor = await vscode.window.showTextDocument(doc);

    // highlight the full chunk range
    const startPos = doc.positionAt(pick.start);
    const endPos = doc.positionAt(pick.end);
    const range = new vscode.Range(startPos, endPos);
    editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
    editor.selection = new vscode.Selection(startPos, endPos);

    // temporary decoration for visual emphasis
    const decoType = vscode.window.createTextEditorDecorationType({ backgroundColor: new vscode.ThemeColor('editor.rangeHighlightBackground') });
    editor.setDecorations(decoType, [range]);
    setTimeout(() => decoType.dispose(), 4000);
  });

  const clearCommand = vscode.commands.registerCommand('semanticSearch.clearIndex', async () => {
    const table = await vectorDB.getTable();
    await table.delete("1=1");
    searchManager.clear();
    vscode.window.showInformationMessage('Semantic index cleared.');
  });

  const statsCommand = vscode.commands.registerCommand('semanticSearch.showStats', async () => {
    const table = await vectorDB.getTable();
    const count = await table.countRows();
    vscode.window.showInformationMessage(`Indexed ${count} chunks`);
  });

  const setApiKeyCommand = vscode.commands.registerCommand('semanticSearch.setApiKey', async () => {
    try {
      const key = await vscode.window.showInputBox({ prompt: 'Paste Google API key (Gemini / Gen AI API)', ignoreFocusOut: true, placeHolder: 'GOOGLE_API_KEY' });
      if (!key) { vscode.window.showInformationMessage('No API key provided.'); return; }
      if (context?.secrets) { await context.secrets.store('GOOGLE_API_KEY', key); vscode.window.showInformationMessage('Stored GOOGLE_API_KEY in VS Code SecretStorage.'); } else { vscode.window.showWarningMessage('Could not access SecretStorage. Set GOOGLE_API_KEY as an env variable for development.'); }
    } catch (err) { vscode.window.showErrorMessage('Failed to store API key: ' + getErrorMessage(err)); }
  });

  context.subscriptions.push(indexCommand, queryCommand, clearCommand, statsCommand, setApiKeyCommand);

  // create file watcher to keep index in sync
  try {
    const fileGlob = vscode.workspace.getConfiguration('semanticSearch').get<string>('fileGlob', '**/*.{js,ts,jsx,tsx,py,java,go,rs,md,html,css}');
    const watcher = vscode.workspace.createFileSystemWatcher(fileGlob, false, false, false);

    watcher.onDidChange(uri => {
      output.appendLine(`File changed: ${uri.fsPath}`);
      scheduleFileUpdate(uri, searchManager, context);
    });
    watcher.onDidCreate(uri => {
      output.appendLine(`File created: ${uri.fsPath}`);
      scheduleFileUpdate(uri, searchManager, context);
    });
    watcher.onDidDelete(async uri => {
      output.appendLine(`File deleted: ${uri.fsPath}`);
      const table = await vectorDB.getTable();
      await table.delete(`uri = "${uri.toString()}"`);
    });

    context.subscriptions.push(watcher);
  } catch (err) {
    output.appendLine('Failed to create file watcher: ' + getErrorMessage(err));
  }

  // create status bar for per-file updates
  fileUpdateStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  context.subscriptions.push(fileUpdateStatus);
}

export function deactivate() { output.appendLine('semantic-code-search: deactivated'); }

function truncate(s: string, n: number) { return s.length <= n ? s : s.slice(0, n - 1) + '…'; }
