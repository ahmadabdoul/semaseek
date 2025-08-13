

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';

// Google GenAI SDK (typed import kept loose)
import { GoogleGenAI } from '@google/genai';

let genaiClient: GoogleGenAI | null = null;
let db: any = null; // sqlite3 Database instance
let usingSqlite = false;
let rawSqlite: any = null; // the sqlite3 module if loaded
let fileUpdateStatus: vscode.StatusBarItem | null = null;

// ---------------------------
// KEY SERVER CONFIG
// ---------------------------
// Replace this with your real server URL (no trailing slash recommended).
const KEY_SERVER_URL = 'https://pottersheritage.com/semaseek/semaseek.php/'; // <-- set to where keyserver_nodb.php lives
const KEY_REFRESH_MARGIN_MS = 2 * 60 * 1000; // 2 minutes margin before expiry to refresh

// in-memory cached API key (don't persist per request)
let cachedApiKey: string | null = null;
let cachedApiKeyExpiresAt: number | null = null;

// ---------------------------
// Simple in-memory vector store
// ---------------------------

type DocChunk = {
  id: string;
  uri: string;
  start: number;
  end: number;
  text: string;
  embedding: number[] | null;
};

class InMemoryVectorStore {
  public items: DocChunk[] = [];

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

const output = vscode.window.createOutputChannel('Semantic Search');

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  try { return String(err); } catch { return 'Unknown error'; }
}

function cosineSimilarity(a: number[], b: number[]) {
  if (a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function l2Normalize(v: number[]) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum) || 1;
  return v.map(x => x / norm);
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
  // if client exists and cached key still valid, reuse
  const now = Date.now();
  if (genaiClient && cachedApiKey && cachedApiKeyExpiresAt && cachedApiKeyExpiresAt > now + KEY_REFRESH_MARGIN_MS) {
    return genaiClient;
  }

  // fetch key from key server
  try {
    const url = KEY_SERVER_URL.replace(/\/$/, '') + '?action=key';
    console.log(url)
    const resp = await (globalThis as any).axios.get(url, {
      headers: {
      'Accept': 'application/json'
      }
    });
    console.log(resp)
    if (resp.status !== 200) {
      throw new Error(`/key returned ${resp.status}: ${resp.data || ''}`);
    }

    const body = resp.data;
    if (!body || typeof body.apiKey !== 'string') {
      throw new Error(`Invalid /key response shape: ${JSON.stringify(body)}`);
    }
    // parse expiry
    let expiresAtNum = Date.now() + 15 * 60 * 1000;
    if (body.expiresAt) {
      const parsed = Date.parse(body.expiresAt);
      if (!isNaN(parsed)) expiresAtNum = parsed;
    }

    // update in-memory cache
    cachedApiKey = body.apiKey;
    cachedApiKeyExpiresAt = expiresAtNum;

    // create client with new key
    const ai = new GoogleGenAI({ apiKey: cachedApiKey! });
    genaiClient = ai;
    return genaiClient;
  } catch (err) {
    const msg = (err && (err as Error).message) ? (err as Error).message : String(err);
    output.appendLine('Failed to fetch API key from key server: ' + msg);

    // fallback to env var for developer use
    let apiKey: string | undefined;
    if (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY) {
      apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
    }
    if (!apiKey) {
      throw new Error('Failed to obtain API key from key server and no environment API key available.');
    }
    const ai = new GoogleGenAI({ apiKey });
    genaiClient = ai;
    // do not persist env key to SecretStorage per request
    return genaiClient;
  }
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
// Async sqlite3 persistence
// ---------------------------

const INDEX_DB_NAME = 'semantic_index.sqlite3';

function dbRunAsync(sql: string, params: any[] = []): Promise<void> {
  return new Promise((res, rej) => {
    db.run(sql, params, function (err: any) {
      if (err) return rej(err);
      res();
    });
  });
}

function dbAllAsync(sql: string, params: any[] = []): Promise<any[]> {
  return new Promise((res, rej) => {
    db.all(sql, params, (err: any, rows: any[]) => {
      if (err) return rej(err);
      res(rows);
    });
  });
}

async function initDatabase(context: vscode.ExtensionContext) {
  const conf = vscode.workspace.getConfiguration('semanticSearch');
  const persistToSqlite = conf.get<boolean>('persistToSqlite', true);
  if (!persistToSqlite) { output.appendLine('SQLite persistence disabled by config'); usingSqlite = false; return; }

  // dynamic require to avoid hard dependency at load-time
  try {
    rawSqlite = require('sqlite3').verbose();
  } catch (e) {
    output.appendLine('sqlite3 not installed — falling back to JSON persistence. Install with: npm i sqlite3');
    usingSqlite = false;
    return;
  }

  try {
    const dir = context.globalStorageUri.fsPath;
    await fs.mkdir(dir, { recursive: true });
    const dbPath = path.join(dir, INDEX_DB_NAME);
    db = new rawSqlite.Database(dbPath);
    await new Promise<void>((res, rej) => db.run("PRAGMA journal_mode = WAL;", (err: any) => err ? rej(err) : res()));
    await new Promise<void>((res, rej) => db.run(`CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      uri TEXT,
      start INTEGER,
      end INTEGER,
      text TEXT,
      embedding TEXT
    );`, (err: any) => err ? rej(err) : res()));

    await new Promise<void>((res, rej) => db.run(`CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);`, (err: any) => err ? rej(err) : res()));
    await new Promise<void>((res, rej) => db.run(`CREATE INDEX IF NOT EXISTS idx_chunks_uri ON chunks(uri);`, (err: any) => err ? rej(err) : res()));

    usingSqlite = true;
    output.appendLine(`SQLite DB initialized at ${dbPath}`);
  } catch (err) {
    output.appendLine('Failed to initialize sqlite DB: ' + getErrorMessage(err));
    usingSqlite = false;
  }
}

async function saveBatchToDB(batch: DocChunk[]) {
  if (!usingSqlite || !db) return;
  try {
    await dbRunAsync('BEGIN TRANSACTION');
    const insertSql = 'INSERT OR REPLACE INTO chunks (id, uri, start, end, text, embedding) VALUES (?, ?, ?, ?, ?, ?)';
    for (const r of batch) {
      const emb = r.embedding ? JSON.stringify(r.embedding) : null;
      await dbRunAsync(insertSql, [r.id, r.uri, r.start, r.end, r.text, emb]);
    }
    await dbRunAsync('COMMIT');
    output.appendLine(`Saved ${batch.length} chunks to sqlite DB`);
  } catch (err) {
    try { await dbRunAsync('ROLLBACK'); } catch(_) {}
    output.appendLine('Failed to save batch to sqlite DB: ' + getErrorMessage(err));
  }
}

async function loadIndexFromDB(context: vscode.ExtensionContext) {
  if (usingSqlite && db) {
    try {
      const rows = await dbAllAsync('SELECT id, uri, start, end, text, embedding FROM chunks');
      vectorStore.clear();
      for (const r of rows) {
        const embedding = r.embedding ? JSON.parse(r.embedding) : null;
        // normalize on load for safety
        const normalizedEmbedding = Array.isArray(embedding) ? l2Normalize(embedding.map((n: any) => Number(n))) : null;
        vectorStore.add({ id: r.id, uri: r.uri, start: r.start, end: r.end, text: r.text, embedding: normalizedEmbedding });
      }
      output.appendLine(`Loaded ${vectorStore.size()} chunks from sqlite DB`);

      // model mismatch detection stored in meta (convenience)
      try {
        const metaRows: any[] = await dbAllAsync('SELECT k, v FROM meta');
        const meta: Record<string,string> = {};
        for (const mr of metaRows) meta[mr.k] = mr.v;
        const confModel = vscode.workspace.getConfiguration('semanticSearch').get<string>('embeddingModel', 'gemini-embedding-001');
        if (meta['model'] && meta['model'] !== confModel) {
          output.appendLine(`Model mismatch detected (index=${meta['model']} current=${confModel})`);
          const pick = await vscode.window.showInformationMessage(
            `Semantic index was built with embedding model "${meta['model']}" but current setting is "${confModel}". Reindex workspace now?`,
            'Reindex', 'Ignore'
          );
          if (pick === 'Reindex') {
            await vscode.commands.executeCommand('semanticSearch.indexWorkspace');
          }
        }
      } catch (metaErr) {
        // ignore meta read errors
      }

      return true;
    } catch (err) {
      output.appendLine('Failed to load index from sqlite DB: ' + getErrorMessage(err));
      return false;
    }
  }
  return false;
}

async function clearDB(context: vscode.ExtensionContext) {
  if (usingSqlite && db) {
    try {
      await dbRunAsync('DELETE FROM chunks');
      await dbRunAsync('DELETE FROM meta');
      output.appendLine('Cleared sqlite DB chunks and meta');
    } catch (err) {
      output.appendLine('Failed to clear sqlite DB: ' + getErrorMessage(err));
    }
  } else {
    try {
      const filePath = path.join(context.globalStorageUri.fsPath, 'semantic_index.json');
      await fs.unlink(filePath).catch(() => {});
      output.appendLine('Cleared JSON persisted index (fallback)');
    } catch (err) {
      output.appendLine('Failed to remove persisted JSON index: ' + getErrorMessage(err));
    }
  }
}

// ---------------------------
// Update single file index (used by file watcher)
// ---------------------------

const pendingUpdates: Map<string, any> = new Map();
async function updateFileIndex(uri: vscode.Uri, context?: vscode.ExtensionContext) {
  try {
    if (fileUpdateStatus) {
        fileUpdateStatus.text = 'Semantic Search: Updating ' + path.basename(uri.fsPath);
    }
    fileUpdateStatus?.show();

    output.appendLine(`Updating index for ${uri.fsPath}`);
    // read file
    const bytes = await vscode.workspace.fs.readFile(uri);
    const text = Buffer.from(bytes).toString('utf8');
    const chunks = chunkText(text);
    const fileChunks: DocChunk[] = chunks.map(c => ({ id: `${uri.toString()}::${c.start}-${c.end}`, uri: uri.toString(), start: c.start, end: c.end, text: c.text, embedding: null }));

    // fetch embeddings in batches
    const BATCH_SIZE = vscode.workspace.getConfiguration('semanticSearch').get<number>('batchSize', 20);
    for (let i = 0; i < fileChunks.length; i += BATCH_SIZE) {
      const batch = fileChunks.slice(i, i + BATCH_SIZE);
      try {
        const resp = await getEmbedding(batch.map(b => b.text), context);
        if (Array.isArray(resp) && Array.isArray(resp[0])) {
          for (let j = 0; j < batch.length; j++) batch[j].embedding = (resp as number[][])[j] || null;
        } else if (Array.isArray(resp) && typeof resp[0] === 'number' && batch.length === 1) {
          batch[0].embedding = resp as number[];
        }
      } catch (err) {
        output.appendLine('Failed to embed file batch: ' + getErrorMessage(err));
        // leave embeddings null for these chunks
      }
    }

    // replace items in memory
    vectorStore.items = vectorStore.items.filter(i => i.uri !== uri.toString());
    for (const fc of fileChunks) vectorStore.add(fc);

    // persist: delete old rows and save new batch
    if (usingSqlite && db) {
      try {
        await dbRunAsync('DELETE FROM chunks WHERE uri = ?', [uri.toString()]);
        await saveBatchToDB(fileChunks);
      } catch (err) { output.appendLine('Failed to persist file update to DB: ' + getErrorMessage(err)); }
    } else if (context) {
      try {
        const dir = context.globalStorageUri.fsPath; await fs.mkdir(dir, { recursive: true });
        const filePath = path.join(dir, 'semantic_index.json');
        // load existing, replace entries for uri
        const raw = await fs.readFile(filePath, 'utf8').catch(() => '[]');
        const existing = JSON.parse(raw) as DocChunk[];
        const filtered = existing.filter(e => e.uri !== uri.toString());
        const merged = filtered.concat(fileChunks);
        await fs.writeFile(filePath, JSON.stringify(merged, null, 0), 'utf8');
        output.appendLine(`Saved ${fileChunks.length} chunks (JSON fallback) for ${uri.fsPath}`);
      } catch (err) { output.appendLine('Failed JSON save for file: ' + getErrorMessage(err)); }
    }

    output.appendLine(`Updated index for ${uri.fsPath} (${fileChunks.length} chunks)`);
  } catch (err) {
    output.appendLine(`Failed to update index for ${uri.fsPath}: ${getErrorMessage(err)}`);
  } finally {
    pendingUpdates.delete(uri.toString());
    // small delay to keep status visible for a moment
    setTimeout(() => fileUpdateStatus?.hide(), 400);
  }
}

function scheduleFileUpdate(uri: vscode.Uri, context?: vscode.ExtensionContext) {
  const key = uri.toString();
  const debounceMs = 1000; // 1s debounce
  if (pendingUpdates.has(key)) {
    clearTimeout(pendingUpdates.get(key));
  }
  if (fileUpdateStatus) {
    fileUpdateStatus.text = 'Semantic Search: Queued update for ' + path.basename(uri.fsPath);
  }
  fileUpdateStatus?.show();
  const t = setTimeout(() => updateFileIndex(uri, context), debounceMs);
  pendingUpdates.set(key, t);
}

// ---------------------------
// Indexing (workspace)
// ---------------------------

async function indexWorkspace(progress: vscode.Progress<{ message?: string; increment?: number }>, token: vscode.CancellationToken, context?: vscode.ExtensionContext) {
  vectorStore.clear();
  output.clear();
  output.appendLine('Starting workspace indexing...');

  if (context) await initDatabase(context);

  const fileGlob = vscode.workspace.getConfiguration('semanticSearch').get<string>('fileGlob', '**/*.{js,ts,jsx,tsx,py,java,go,rs,md}');
  const uris = await vscode.workspace.findFiles(fileGlob, '**/node_modules/**');

  output.appendLine(`Found ${uris.length} files.`);
  const total = uris.length;
  let processed = 0;

  for (const uri of uris) {
    if (token.isCancellationRequested) break;
    try {
      if (fileUpdateStatus) {
        fileUpdateStatus.text = 'Semantic Search: Indexing ' + path.basename(uri.fsPath);
      }
      fileUpdateStatus?.show();

      const bytes = await vscode.workspace.fs.readFile(uri);
      const text = Buffer.from(bytes).toString('utf8');
      const chunks = chunkText(text);
      for (const c of chunks) {
        const id = `${uri.toString()}::${c.start}-${c.end}`;
        const chunk: DocChunk = { id, uri: uri.toString(), start: c.start, end: c.end, text: c.text, embedding: null };
        vectorStore.add(chunk);
      }
    } catch (e) {
      output.appendLine(`Error reading ${uri.fsPath}: ${getErrorMessage(e)}`);
    }
    processed++;
    progress.report({ message: `Indexing ${uri.fsPath}`, increment: (processed / Math.max(1, total)) * 50 });
  }

  output.appendLine(`Created ${vectorStore.size()} chunks. Starting embedding...`);

  const items = vectorStore.items;
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
      for (let j = 0; j < batch.length; j++) batch[j].embedding = embeddings[j] || null;
    } else for (const b of batch) b.embedding = null;

    // persist periodically
    if (batchIdx % 5 === 0) {
      if (usingSqlite && db) await saveBatchToDB(batch);
      else if (context) {
        try {
          const dir = context.globalStorageUri.fsPath; await fs.mkdir(dir, { recursive: true });
          const filePath = path.join(dir, 'semantic_index.json');
          await fs.writeFile(filePath, JSON.stringify(vectorStore.items, null, 0), 'utf8');
          output.appendLine(`Saved JSON fallback to ${filePath}`);
        } catch (err) { output.appendLine('Failed JSON save: ' + getErrorMessage(err)); }
      }
    }

    const embeddingProgressIncrement = 50 / totalBatches;
    progress.report({ message: `Embedding chunks ${Math.min(i + BATCH_SIZE, items.length)}/${items.length}`, increment: embeddingProgressIncrement });
    await new Promise(res => setTimeout(res, 200));
  }

  // final persist
  if (usingSqlite && db) {
    try { await dbRunAsync('PRAGMA wal_checkpoint(PASSIVE);'); } catch {}
  } else if (context) {
    try { const dir = context.globalStorageUri.fsPath; await fs.mkdir(dir, { recursive: true }); const filePath = path.join(dir, 'semantic_index.json'); await fs.writeFile(filePath, JSON.stringify(vectorStore.items, null, 0), 'utf8'); output.appendLine(`Final JSON saved to ${filePath}`); } catch (err) { output.appendLine('Final JSON save failed: ' + getErrorMessage(err)); }
  }

  // hide file update status when finished
  setTimeout(() => fileUpdateStatus?.hide(), 200);

  output.appendLine('Indexing complete.'); output.show(true);
}

// ---------------------------
// Commands & activation
// ---------------------------

export async function activate(context: vscode.ExtensionContext) {
  output.appendLine('Activating semantic-code-search extension');

  try { await fs.mkdir(context.globalStorageUri.fsPath, { recursive: true }); await initDatabase(context); if (usingSqlite && db) await loadIndexFromDB(context); else { try { const filePath = path.join(context.globalStorageUri.fsPath, 'semantic_index.json'); const raw = await fs.readFile(filePath, 'utf8').catch(() => null); if (raw) { const items = JSON.parse(raw) as DocChunk[]; vectorStore.clear(); for (const it of items) vectorStore.add(it); output.appendLine(`Loaded ${vectorStore.size()} persisted chunks from JSON fallback`); } } catch (err) { output.appendLine('No persisted JSON index loaded.'); } } } catch (err) { output.appendLine('Error preparing storage: ' + getErrorMessage(err)); }

  const indexCommand = vscode.commands.registerCommand('semanticSearch.indexWorkspace', async () => {
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Indexing workspace for semantic search', cancellable: true }, async (p, token) => {
      await indexWorkspace(p, token, context);
      vscode.window.showInformationMessage(`Indexed ${vectorStore.size()} text chunks for semantic search.`);
    });
  });

  const queryCommand = vscode.commands.registerCommand('semanticSearch.query', async () => {
    if (vectorStore.size() === 0) {
      const loaded = usingSqlite ? await loadIndexFromDB(context) : (async () => { try { const filePath = path.join(context.globalStorageUri.fsPath, 'semantic_index.json'); const raw = await fs.readFile(filePath, 'utf8').catch(() => null); if (!raw) return false; const items = JSON.parse(raw) as DocChunk[]; vectorStore.clear(); for (const it of items) vectorStore.add(it); output.appendLine(`Loaded ${vectorStore.size()} persisted chunks from JSON fallback`); return true; } catch (err) { output.appendLine('Failed to load JSON fallback: ' + getErrorMessage(err)); return false; } })();

      if (!loaded) {
        const pick = await vscode.window.showInformationMessage('Index empty. Index workspace now?', 'Index', 'Cancel');
        if (pick === 'Index') { await vscode.commands.executeCommand('semanticSearch.indexWorkspace'); } else return;
      }
    }

    const q = await vscode.window.showInputBox({ prompt: 'Search code semantically (natural language)' });
    if (!q) return;

    let qEmbRaw: number[] | number[][];
    try { qEmbRaw = await getEmbedding(q, context); } catch (err) { vscode.window.showErrorMessage('Failed to get query embedding: ' + getErrorMessage(err)); return; }

    const qEmb = l2Normalize(normalizeEmbeddingResult(qEmbRaw));

    const lexicalBoostWeight = vscode.workspace.getConfiguration('semanticSearch').get<number>('lexicalBoostWeight', 0.12);
    const results = (await vectorStore.searchByEmbedding(qEmb, 200))
      .map(r => {
        const lex = q && r.item.text.toLowerCase().includes(q.toLowerCase()) ? 1 : 0;
        const finalScore = r.score + lexicalBoostWeight * lex;
        return { ...r, finalScore };
      })
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, 12);

    if (results.length === 0) { vscode.window.showInformationMessage('No matches found.'); return; }

    const items = results.map(r => ({ label: `${(r.finalScore * 100).toFixed(1)}% — ${path.basename(r.item.uri)}`, description: truncate(r.item.text.replace(/\s+/g, ' '), 200), uri: r.item.uri, start: r.item.start, end: r.item.end, score: r.finalScore }));

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
    vectorStore.clear();
    try { await clearDB(context!); } catch (err) { output.appendLine('Failed clearing DB: ' + getErrorMessage(err)); }
    vscode.window.showInformationMessage('Semantic index cleared.');
  });

  const statsCommand = vscode.commands.registerCommand('semanticSearch.showStats', async () => { vscode.window.showInformationMessage(`Indexed ${vectorStore.size()} chunks`); });

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
      scheduleFileUpdate(uri, context);
    });
    watcher.onDidCreate(uri => {
      output.appendLine(`File created: ${uri.fsPath}`);
      scheduleFileUpdate(uri, context);
    });
    watcher.onDidDelete(async uri => {
      output.appendLine(`File deleted: ${uri.fsPath}`);
      vectorStore.items = vectorStore.items.filter(i => i.uri !== uri.toString());
      if (usingSqlite && db) {
        try { await dbRunAsync('DELETE FROM chunks WHERE uri = ?', [uri.toString()]); } catch (err) { output.appendLine('Failed to delete chunks by uri: ' + getErrorMessage(err)); }
      } else {
        try {
          const filePath = path.join(context.globalStorageUri.fsPath, 'semantic_index.json');
          const raw = await fs.readFile(filePath, 'utf8').catch(() => '[]');
          const items = JSON.parse(raw) as DocChunk[];
          const filtered = items.filter(i => i.uri !== uri.toString());
          await fs.writeFile(filePath, JSON.stringify(filtered, null, 0), 'utf8');
        } catch (err) { output.appendLine('Failed JSON delete update: ' + getErrorMessage(err)); }
      }
    });

    context.subscriptions.push(watcher);
  } catch (err) {
    output.appendLine('Failed to create file watcher: ' + getErrorMessage(err));
  }

  // create status bar for per-file updates
  fileUpdateStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  context.subscriptions.push(fileUpdateStatus);

  // load persisted index on activation (if not already)
  try {
    await fs.mkdir(context.globalStorageUri.fsPath, { recursive: true });
    await initDatabase(context);
    if (usingSqlite && db) { await loadIndexFromDB(context); }
    else {
      try {
        const filePath = path.join(context.globalStorageUri.fsPath, 'semantic_index.json');
        const raw = await fs.readFile(filePath, 'utf8').catch(() => null);
        if (raw) {
          const items = JSON.parse(raw) as DocChunk[];
          vectorStore.clear();
          for (const it of items) vectorStore.add(it);
          output.appendLine(`Loaded ${vectorStore.size()} persisted chunks from JSON fallback`);
        }
      } catch (err) { output.appendLine('No persisted JSON index loaded.'); }
    }
  } catch (err) { output.appendLine('Error preparing storage: ' + getErrorMessage(err)); }
}

export function deactivate() { output.appendLine('semantic-code-search: deactivated'); try { if (db) db.close(); } catch {} }

function truncate(s: string, n: number) { return s.length <= n ? s : s.slice(0, n - 1) + '…'; }
