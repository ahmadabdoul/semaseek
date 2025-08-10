/**
  extension.reindex-prompt.ts - semantic-code-search (full, day-1 starter)

  - Uses Gemini embeddings via @google/genai (configurable model)
  - L2-normalizes embeddings for stable cosine search
  - Persists to async sqlite3 with JSON fallback
  - Stores meta (model + dim) and prompts to Reindex when mismatch detected
  - Commands: indexWorkspace, query, clearIndex, showStats, setApiKey

  NOTES: add configuration entries to package.json (see bottom comment).
*/

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import { GoogleGenAI } from '@google/genai';

let genaiClient: GoogleGenAI | null = null;
let db: any = null; // sqlite3 Database instance
let usingSqlite = false;
let rawSqlite: any = null; // sqlite3 module

// ---------------------------
// Config helpers
// ---------------------------
function getConfig() {
  const conf = vscode.workspace.getConfiguration('semanticSearch');
  return {
    persistToSqlite: conf.get<boolean>('persistToSqlite', true),
    embeddingModel: conf.get<string>('embeddingModel', 'gemini-embedding-001'),
    batchSize: conf.get<number>('batchSize', 20),
    saveEveryBatches: conf.get<number>('saveEveryBatches', 5),
    lexicalBoostWeight: conf.get<number>('lexicalBoostWeight', 0.12)
  };
}

// ---------------------------
// In-memory vector store
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

  add(chunk: DocChunk) { this.items.push(chunk); }
  clear() { this.items = []; }
  size() { return this.items.length; }

  // simple cosine similarity search
  async searchByEmbedding(qEmb: number[], topK = 10) {
    const scored = this.items
      .filter(i => i.embedding && i.embedding.length === qEmb.length)
      .map(i => ({ item: i, score: cosineSimilarity(i.embedding! as number[], qEmb) }))
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
// GenAI (Gemini) client + embedding
// ---------------------------

async function initGenAI(context?: vscode.ExtensionContext) {
  if (genaiClient) return genaiClient;
  let apiKey: string | undefined;
  try { if (context?.secrets) apiKey = (await context.secrets.get('GOOGLE_API_KEY')) || undefined; } catch {}
  if (!apiKey && (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY)) apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('Google API key not found. Use command "Semantic Search: Set API Key" or set env GOOGLE_API_KEY.');
  const ai = new GoogleGenAI({ apiKey });
  genaiClient = ai;
  return genaiClient;
}

/**
 * getEmbedding: uses configured model (defaults to textembedding-gecko@003)
 * Returns normalized vectors (number[] or number[][])
 */
export async function getEmbedding(text: string | string[], context?: vscode.ExtensionContext): Promise<number[] | number[][]> {
  const client = await initGenAI(context);
  const contents = Array.isArray(text) ? text : [text];
  const MODEL = getConfig().embeddingModel;
  const TASK = 'SEMANTIC_SIMILARITY';

  try {
    // call embedContent per Gemini docs — SDK shapes vary between versions, parse defensively
    const resp: any = await (client as any).models.embedContent({ model: MODEL, contents, taskType: TASK });

    // common shape: resp.embeddings -> [{values: [...]}, ...]
    if (Array.isArray(resp?.embeddings) && resp.embeddings.length > 0) {
      const extracted = resp.embeddings.map((e: any) => {
        if (Array.isArray(e?.values)) return e.values as number[];
        if (Array.isArray(e?.embedding?.values)) return e.embedding.values as number[];
        if (Array.isArray(e?.embedding)) return e.embedding as number[];
        if (Array.isArray(e)) return e as number[];
        throw new Error('Unexpected embedding entry shape');
      });
      const normalized = extracted.map((v: any[]) => l2Normalize(v.map(n => Number(n))));
      return Array.isArray(text) ? normalized : normalized[0];
    }

    // fallback: resp.data -> [{embedding: {values: [...]}}]
    if (Array.isArray(resp?.data) && resp.data.length > 0) {
      const extracted2 = resp.data.map((d: any) => {
        if (Array.isArray(d?.embedding?.values)) return d.embedding.values as number[];
        if (Array.isArray(d?.embedding)) return d.embedding as number[];
        if (Array.isArray(d)) return d as number[];
        throw new Error('Unexpected data.embedding shape');
      });
      const normalized = extracted2.map((v: any[]) => l2Normalize(v.map(n => Number(n))));
      return Array.isArray(text) ? normalized : normalized[0];
    }

    // last attempt: resp itself is a vector
    if (!Array.isArray(text) && Array.isArray(resp) && typeof resp[0] === 'number') {
      return l2Normalize((resp as number[]).map(n => Number(n)));
    }

    throw new Error('Unexpected embeddings response shape');
  } catch (err) {
    throw new Error(`Gemini embedContent failed: ${getErrorMessage(err)}`);
  }
}

// ---------------------------
// Async sqlite3 persistence (with meta)
// ---------------------------

const INDEX_DB_NAME = 'semantic_index.sqlite3';

function dbRunAsync(sql: string, params: any[] = []): Promise<void> {
  return new Promise((res, rej) => {
    db.run(sql, params, function (err: any) { if (err) return rej(err); res(); });
  });
}

function dbAllAsync(sql: string, params: any[] = []): Promise<any[]> {
  return new Promise((res, rej) => {
    db.all(sql, params, (err: any, rows: any[]) => { if (err) return rej(err); res(rows); });
  });
}

async function initDatabase(context: vscode.ExtensionContext) {
  const conf = getConfig();
  if (!conf.persistToSqlite) {
    output.appendLine('SQLite persistence disabled by config');
    usingSqlite = false; return;
  }

  try { rawSqlite = require('sqlite3').verbose(); } catch (e) {
    output.appendLine('sqlite3 not installed — falling back to JSON persistence. Install with: npm i sqlite3');
    usingSqlite = false; return;
  }

  try {
    const dir = context.globalStorageUri.fsPath; await fs.mkdir(dir, { recursive: true });
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

    usingSqlite = true; output.appendLine(`SQLite DB initialized at ${dbPath}`);
  } catch (err) { output.appendLine('Failed to initialize sqlite DB: ' + getErrorMessage(err)); usingSqlite = false; }
}

async function setMeta(key: string, value: string) { if (!usingSqlite || !db) return; await dbRunAsync('INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)', [key, value]); }
async function getMeta(key: string): Promise<string | null> { if (!usingSqlite || !db) return null; const rows = await dbAllAsync('SELECT v FROM meta WHERE k = ?', [key]); if (!rows || rows.length === 0) return null; return rows[0].v; }

async function saveBatchToDB(batch: DocChunk[], context?: vscode.ExtensionContext) {
  if (!usingSqlite || !db) return;
  try {
    await dbRunAsync('BEGIN TRANSACTION');
    const insertSql = 'INSERT OR REPLACE INTO chunks (id, uri, start, end, text, embedding) VALUES (?, ?, ?, ?, ?, ?)';
    await new Promise<void>((resolve, reject) => {
      const stmt = db.prepare(insertSql);
      let i = 0;
      function runNext() {
        if (i >= batch.length) { stmt.finalize((err: any) => err ? reject(err) : resolve()); return; }
        const r = batch[i++];
        const emb = r.embedding ? JSON.stringify(r.embedding) : null;
        stmt.run([r.id, r.uri, r.start, r.end, r.text, emb], (err: any) => { if (err) return reject(err); runNext(); });
      }
      runNext();
    });
    await dbRunAsync('COMMIT');

    // save meta if missing
    const modelMeta = await getMeta('model');
    if (!modelMeta && context) {
      const conf = getConfig();
      const firstDim = batch.find(b => Array.isArray(b.embedding) && b.embedding.length > 0)?.embedding?.length ?? null;
      if (firstDim) { await setMeta('model', conf.embeddingModel); await setMeta('dim', String(firstDim)); output.appendLine(`Saved meta model=${conf.embeddingModel} dim=${firstDim}`); }
    }

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
      let nullEmbCount = 0;
      for (const r of rows) {
        const embedding = r.embedding ? JSON.parse(r.embedding) : null;
        if (!embedding) nullEmbCount++;
        const normalizedEmbedding = Array.isArray(embedding) ? l2Normalize(embedding.map((n: any) => Number(n))) : null;
        vectorStore.add({ id: r.id, uri: r.uri, start: r.start, end: r.end, text: r.text, embedding: normalizedEmbedding });
      }
      output.appendLine(`Loaded ${vectorStore.size()} chunks from sqlite DB (null embeddings: ${nullEmbCount})`);

      // check meta
      const modelMeta = await getMeta('model');
      const dimMeta = await getMeta('dim');
      const conf = getConfig();
      if (modelMeta && modelMeta !== conf.embeddingModel) {
        output.appendLine(`Model mismatch detected (index=${modelMeta} current=${conf.embeddingModel}). Prompting user to reindex.`);
        const pick = await vscode.window.showInformationMessage(
          `Semantic index was built with embedding model "${modelMeta}" but current setting is "${conf.embeddingModel}". Reindex workspace now?`,
          'Reindex', 'Ignore'
        );
        if (pick === 'Reindex') {
          try { await vscode.commands.executeCommand('semanticSearch.indexWorkspace'); } catch (err) { output.appendLine('Failed to start reindex: ' + getErrorMessage(err)); }
        }
      }

      if (dimMeta) {
        const dimNum = Number(dimMeta);
        const firstEmb = vectorStore.items.find(i => Array.isArray(i.embedding) && i.embedding.length > 0)?.embedding;
        if (firstEmb && firstEmb.length !== dimNum) {
          output.appendLine(`Warning: meta dim=${dimNum} but first embedding length=${firstEmb.length}`);
        }
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
    try { await dbRunAsync('DELETE FROM chunks'); await dbRunAsync('DELETE FROM meta'); output.appendLine('Cleared sqlite DB chunks and meta'); } catch (err) { output.appendLine('Failed to clear sqlite DB: ' + getErrorMessage(err)); }
  } else {
    try { const filePath = path.join(context.globalStorageUri.fsPath, 'semantic_index.json'); await fs.unlink(filePath).catch(() => {}); output.appendLine('Cleared JSON persisted index (fallback)'); } catch (err) { output.appendLine('Failed to remove persisted JSON index: ' + getErrorMessage(err)); }
  }
}

// ---------------------------
// Indexing
// ---------------------------

async function indexWorkspace(progress: vscode.Progress<{ message?: string; increment?: number }>, token: vscode.CancellationToken, context?: vscode.ExtensionContext) {
  vectorStore.clear(); output.clear(); output.appendLine('Starting workspace indexing...');
  if (!context) throw new Error('Extension context required');
  await initDatabase(context);

  const fileGlob = '**/*.{js,ts,jsx,tsx,py,java,go,rs,md}';
  const uris = await vscode.workspace.findFiles(fileGlob, '**/node_modules/**');
  output.appendLine(`Found ${uris.length} files.`);
  const total = uris.length; let processed = 0;

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
    } catch (e) { output.appendLine(`Error reading ${uri.fsPath}: ${getErrorMessage(e)}`); }
    processed++;
    progress.report({ message: `Indexing ${uri.fsPath}`, increment: (processed / Math.max(1, total)) * 50 });
  }

  output.appendLine(`Created ${vectorStore.size()} chunks. Starting embedding...`);

  const items = vectorStore.items;
  const conf = getConfig();
  const BATCH_SIZE = conf.batchSize;
  const SAVE_EVERY = conf.saveEveryBatches;
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
        attempt++; output.appendLine(`Embedding batch ${batchIdx + 1}/${totalBatches} (attempt ${attempt})`);
        const resp = await getEmbedding(texts, context);
        if (Array.isArray(resp) && Array.isArray(resp[0])) { embeddings = resp as number[][]; break; }
        throw new Error('Unexpected embedding shape');
      } catch (err) {
        const msg = getErrorMessage(err); output.appendLine(`Error embedding batch ${batchIdx + 1}: ${msg}`);
        if (attempt >= maxAttempts) { output.appendLine(`Giving up on batch ${batchIdx + 1}`); break; }
        const backoffMs = Math.pow(2, attempt) * 500 + Math.floor(Math.random() * 500);
        output.appendLine(`Retrying in ${backoffMs}ms...`);
        await new Promise(res => setTimeout(res, backoffMs));
      }
    }

    if (embeddings) for (let j = 0; j < batch.length; j++) batch[j].embedding = Array.isArray(embeddings[j]) ? l2Normalize(embeddings[j].map(n => Number(n))) : null;
    else for (const b of batch) b.embedding = null;

    if (batchIdx % SAVE_EVERY === 0) {
      if (usingSqlite && db) await saveBatchToDB(batch, context);
      else if (context) {
        try { const dir = context.globalStorageUri.fsPath; await fs.mkdir(dir, { recursive: true }); const filePath = path.join(dir, 'semantic_index.json'); await fs.writeFile(filePath, JSON.stringify(vectorStore.items, null, 0), 'utf8'); output.appendLine(`Saved JSON fallback to ${filePath}`); } catch (err) { output.appendLine('Failed JSON save: ' + getErrorMessage(err)); }
      }
    }

    const embeddingProgressIncrement = 50 / totalBatches;
    progress.report({ message: `Embedding chunks ${Math.min(i + BATCH_SIZE, items.length)}/${items.length}`, increment: embeddingProgressIncrement });
    await new Promise(res => setTimeout(res, 200));
  }

  if (usingSqlite && db) { try { await dbRunAsync('PRAGMA wal_checkpoint(PASSIVE);'); } catch {} }
  else if (context) { try { const dir = context.globalStorageUri.fsPath; await fs.mkdir(dir, { recursive: true }); const filePath = path.join(dir, 'semantic_index.json'); await fs.writeFile(filePath, JSON.stringify(vectorStore.items, null, 0), 'utf8'); output.appendLine(`Final JSON saved to ${filePath}`); } catch (err) { output.appendLine('Final JSON save failed: ' + getErrorMessage(err)); } }

  output.appendLine('Indexing complete.'); output.show(true);
}

// ---------------------------
// Commands & activation
// ---------------------------

export async function activate(context: vscode.ExtensionContext) {
  output.appendLine('Activating semantic-code-search extension');

  // Register commands first so reindex prompt can call them
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

    const lexicalBoostWeight = getConfig().lexicalBoostWeight;
    const results = (await vectorStore.searchByEmbedding(qEmb, 200))
      .map(r => {
        const lex = q && r.item.text.toLowerCase().includes(q.toLowerCase()) ? 1 : 0;
        const finalScore = r.score + lexicalBoostWeight * lex;
        return { ...r, finalScore };
      })
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, 12);

    if (results.length === 0) { vscode.window.showInformationMessage('No matches found.'); return; }

    const items = results.map(r => ({ label: `${(r.finalScore * 100).toFixed(1)}% — ${path.basename(r.item.uri)}`, description: truncate(r.item.text.replace(/\s+/g, ' '), 200), uri: r.item.uri, start: r.item.start, score: r.finalScore }));

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

  // load persisted index and prompt to reindex when necessary
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

