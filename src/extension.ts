/**
  semantic-code-search - patched VS Code extension

  - Uses Gemini embedContent per docs: https://ai.google.dev/gemini-api/docs/embeddings
  - Robust batching, retries, jitter, and logging
  - Set API key stored in VS Code SecretStorage
*/

import * as vscode from 'vscode';
import * as path from 'path';
import { GoogleGenAI } from '@google/genai';

let genaiClient: GoogleGenAI | null = null;

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

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  try {
    return String(err);
  } catch {
    return 'Unknown error';
  }
}

function cosineSimilarity(a: number[], b: number[]) {
  if (a.length !== b.length) return 0;
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

/**
 * Normalize embedding result:
 * getEmbedding may return number[] (single) or number[][] (batch).
 * This returns a single vector (first) for callers that expect one vector.
 */
function normalizeEmbeddingResult(res: number[] | number[][]): number[] {
  if (Array.isArray(res) && res.length > 0 && Array.isArray(res[0])) {
    return (res as number[][])[0];
  }
  return res as number[];
}

// ---------------------------
// Gemini / GenAI embedding adapter (per docs)
// ---------------------------

async function initGenAI(context?: vscode.ExtensionContext) {
  if (genaiClient) return genaiClient;

  // try SecretStorage (recommended)
  let apiKey: string | undefined;
  try {
    if (context?.secrets) {
      apiKey = (await context.secrets.get('GOOGLE_API_KEY')) || undefined;
    }
  } catch {
    apiKey = undefined;
  }

  // fallback to env var (local dev)
  if (!apiKey && (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY)) {
    apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  }

  if (!apiKey) {
    throw new Error(
      'Google API key not found. Store one in VS Code SecretStorage under key GOOGLE_API_KEY, or set env GOOGLE_API_KEY/GEMINI_API_KEY.'
    );
  }

  // Construct the GoogleGenAI client (SDK expects constructor with options)
  const ai = new GoogleGenAI({ apiKey });
  genaiClient = ai;
  return genaiClient;
}

/**
 * getEmbedding: uses client.models.embedContent per Gemini docs.
 * Accepts either a single string or an array of strings.
 * Returns number[] for single input, number[][] for batch.
 */
export async function getEmbedding(text: string | string[], context?: vscode.ExtensionContext): Promise<number[] | number[][]> {
  const client = await initGenAI(context);
  const contents = Array.isArray(text) ? text : [text];

  // Use gemini model per docs. Could be switched to textembedding-gecko@003 if wanted.
  const MODEL = 'gemini-embedding-001';
  const TASK = 'SEMANTIC_SIMILARITY';

  try {
    // According to docs: response.embeddings is the array, each entry has `.values`
    const resp: any = await (client as any).models.embedContent({
      model: MODEL,
      contents,          // array of strings
      taskType: TASK
      // some SDK versions use `taskType` or `config: { taskType: ... }`
    });

    // resp.embeddings is expected (per docs examples)
    // defensive extraction:
    if (Array.isArray(resp?.embeddings) && resp.embeddings.length > 0) {
      // embeddings entries may be objects that include `.values`
      const extracted: number[][] = resp.embeddings.map((e: any) => {
        if (Array.isArray(e?.values)) return e.values as number[];
        // some shapes may include `value` or `embedding` — try alternatives
        if (Array.isArray(e?.embedding?.values)) return e.embedding.values as number[];
        if (Array.isArray(e?.embedding)) return e.embedding as number[];
        if (Array.isArray(e?.values)) return e.values as number[];
        // fallback: if e is already an array
        if (Array.isArray(e)) return e as number[];
        // if nothing matches, throw to be caught below
        throw new Error('Unexpected embedding entry shape');
      });
      return Array.isArray(text) ? extracted : extracted[0];
    }

    // Some SDK variants return `embeddings` nested differently (defensive)
    // try resp.data -> map to .embedding.values
    if (Array.isArray(resp?.data) && resp.data.length > 0) {
      const extracted2: number[][] = resp.data.map((d: any) => {
        if (Array.isArray(d?.embedding?.values)) return d.embedding.values as number[];
        if (Array.isArray(d?.embedding)) return d.embedding as number[];
        if (Array.isArray(d?.values)) return d.values as number[];
        throw new Error('Unexpected data.embedding shape');
      });
      return Array.isArray(text) ? extracted2 : extracted2[0];
    }

    // Last resort: resp itself is an array of numbers for single input
    if (!Array.isArray(text) && Array.isArray(resp) && typeof resp[0] === 'number') {
      return resp as number[];
    }

    throw new Error('Unexpected Gemini embeddings response shape — check SDK docs or log full response.');
  } catch (err) {
    // surface helpful error with some context
    throw new Error(`Gemini embedContent failed: ${getErrorMessage(err)}`);
  }
}

// ---------------------------
// Indexing logic with robust retry/backoff/jitter + logging
// ---------------------------

const output = vscode.window.createOutputChannel('Semantic Search');

async function indexWorkspace(
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  token: vscode.CancellationToken,
  context?: vscode.ExtensionContext
) {
  vectorStore.clear();
  output.clear();
  output.appendLine('Starting workspace indexing...');

  const fileGlob = '**/*.{js,ts,jsx,tsx,py,java,go,rs,md}';
  const uris = await vscode.workspace.findFiles(fileGlob, '**/node_modules/**');

  output.appendLine(`Found ${uris.length} files matching pattern.`);
  const total = uris.length;
  let processed = 0;

  // Build chunks
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
      output.appendLine(`Error reading file ${uri.fsPath}: ${getErrorMessage(e)}`);
    }
    processed++;
    // progress increment logic: cap to 50% for scanning phase
    progress.report({ message: `Indexing ${uri.fsPath}`, increment: (processed / Math.max(1, total)) * 50 });
  }

  output.appendLine(`Created ${vectorStore.size()} chunks. Starting embedding...`);

  const items = vectorStore.items;
  const BATCH_SIZE = 20; // safer batch size per typical rate limits
  const maxAttempts = 6;

  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    if (token.isCancellationRequested) break;

    const batch = items.slice(i, i + BATCH_SIZE);
    const texts = batch.map(b => b.text);

    let embeddings: number[][] | null = null;
    let attempt = 0;

    while (attempt < maxAttempts) {
      try {
        attempt++;
        output.appendLine(`Embedding batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(items.length / BATCH_SIZE)} (attempt ${attempt})`);
        const resp = await getEmbedding(texts, context); // will return number[][] on success
        if (Array.isArray(resp) && Array.isArray(resp[0])) {
          embeddings = resp as number[][];
          break;
        } else {
          throw new Error('Unexpected embedding return shape (not number[][])');
        }
      } catch (err) {
        const msg = getErrorMessage(err);
        output.appendLine(`Error embedding batch ${Math.floor(i / BATCH_SIZE) + 1}: ${msg}`);
        // If we hit final attempt, record and continue
        if (attempt >= maxAttempts) {
          output.appendLine(`Giving up on batch ${Math.floor(i / BATCH_SIZE) + 1} after ${attempt} attempts.`);
          break;
        }
        // Determine backoff: exponential * 500ms + jitter
        const backoffMs = Math.pow(2, attempt) * 500 + Math.floor(Math.random() * 500);
        output.appendLine(`Retrying in ${backoffMs}ms...`);
        await new Promise(res => setTimeout(res, backoffMs));
      }
    }

    // Attach embeddings (if available) otherwise null
    if (embeddings) {
      for (let j = 0; j < batch.length; j++) {
        batch[j].embedding = embeddings[j] || null;
      }
    } else {
      for (const b of batch) b.embedding = null;
    }

    // report progress for embedding phase (remaining 50% split across batches)
    const batchesTotal = Math.max(1, Math.ceil(items.length / BATCH_SIZE));
    const currentBatchIndex = Math.floor(i / BATCH_SIZE);
    const embeddingProgressIncrement = 50 / batchesTotal;
    progress.report({
      message: `Embedding chunks ${Math.min(i + BATCH_SIZE, items.length)}/${items.length}`,
      increment: embeddingProgressIncrement
    });

    // throttle between batches to reduce bursts
    await new Promise(res => setTimeout(res, 200));
  }

  output.appendLine('Indexing complete.');
  output.show(true);
}

// ---------------------------
// Commands & activation
// ---------------------------

export async function activate(context: vscode.ExtensionContext) {
  output.appendLine('Activating semantic-code-search extension');

  const indexCommand = vscode.commands.registerCommand('semanticSearch.indexWorkspace', async () => {
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Indexing workspace for semantic search', cancellable: true }, async (p, token) => {
      await indexWorkspace(p, token, context);
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

    // get embedding (may return single vector or batch)
    let qEmbRaw: number[] | number[][];
    try {
      qEmbRaw = await getEmbedding(q, context);
    } catch (err) {
      vscode.window.showErrorMessage('Failed to get query embedding: ' + getErrorMessage(err));
      return;
    }

    const qEmb = normalizeEmbeddingResult(qEmbRaw);
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

  // Set / Store Google API key in VS Code SecretStorage
  const setApiKeyCommand = vscode.commands.registerCommand('semanticSearch.setApiKey', async () => {
    try {
      const key = await vscode.window.showInputBox({
        prompt: 'Paste Google API key (Gemini / Gen AI API)',
        ignoreFocusOut: true,
        placeHolder: 'GOOGLE_API_KEY'
      });
      if (!key) {
        vscode.window.showInformationMessage('No API key provided.');
        return;
      }

      if (context?.secrets) {
        await context.secrets.store('GOOGLE_API_KEY', key);
        vscode.window.showInformationMessage('Stored GOOGLE_API_KEY in VS Code SecretStorage.');
      } else {
        vscode.window.showWarningMessage('Could not access SecretStorage. Set GOOGLE_API_KEY as an env variable for development.');
      }
    } catch (err) {
      vscode.window.showErrorMessage('Failed to store API key: ' + getErrorMessage(err));
    }
  });

  context.subscriptions.push(indexCommand, queryCommand, clearCommand, statsCommand, setApiKeyCommand);
}

export function deactivate() {
  output.appendLine('semantic-code-search: deactivated');
}

// small helpers
function truncate(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

