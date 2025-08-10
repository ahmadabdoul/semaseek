/**
  semantic-code-search - starter VS Code extension (single-file starter)

  - workspace indexing command (walks files, chunks, creates embeddings via Gemini)
  - in-memory vector store with cosine-similarity search
  - query command (input box -> semantic search -> quickpick results -> open editor at match)
  - stats, clear-index, and set-api-key commands
*/

import * as vscode from 'vscode';
import * as path from 'path';
// dynamic import of the GenAI SDK types (we'll import the runtime below)
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
  // expose items for indexing code (small helper)
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
// Gemini / GenAI embedding adapter
// ---------------------------

/**
 * Initialize GenAI client (one-time).
 * Priority order for API key:
 *  1) context.secrets (recommended for installed extension)
 *  2) process.env.GOOGLE_API_KEY or process.env.GEMINI_API_KEY (dev or CI)
 */
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

  // Construct the GoogleGenAI client
  // (typed import at top; actual runtime is available after npm i @google/genai)
  const ai = new GoogleGenAI({ apiKey });
  genaiClient = ai;
  return genaiClient;
}

/**
 * Generate embeddings for a single string or a batch of strings.
 * Returns number[] for single input, or number[][] for multiple inputs.
 */
export async function getEmbedding(text: string | string[], context?: vscode.ExtensionContext): Promise<number[] | number[][]> {
  const client = await initGenAI(context);

  const contents = Array.isArray(text) ? text : [text];
  const MODEL = 'gemini-embedding-001';

  try {
    // NOTE: shape and method names depend on SDK version. `client.models.embedContent` is used in examples.
    // Use `as any` to avoid tight TypeScript coupling to SDK shapes.
    const resp: any = await (client as any).models.embedContent({
      model: MODEL,
      contents,
      // optional: config: { outputDimensionality: 1536 }
    });

    // Defensive extraction of embeddings from common SDK shapes
    const tryExtract = (r: any): number[][] | null => {
      if (!r) return null;
      // common: r.embeddings -> [[...], [...]]
      if (Array.isArray(r.embeddings) && r.embeddings.length > 0 && Array.isArray(r.embeddings[0])) {
        return r.embeddings as number[][];
      }
      // common: r.data -> [{ embedding: [...] }, ...] or embedding.values
      if (Array.isArray(r.data) && r.data.length > 0) {
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
      // some SDKs use outputs
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
      throw new Error('Unexpected Gemini embeddings response shape — check SDK docs or log full response.');
    }

    if (!Array.isArray(text)) return extracted[0];
    return extracted;
  } catch (err) {
    throw new Error(`Gemini embedContent failed: ${getErrorMessage(err)}`);
  }
}

// ---------------------------
// Indexing logic
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
    progress.report({ message: `Indexing ${uri.fsPath}`, increment: (processed / total) * 50 });
  }

  output.appendLine(`Created ${vectorStore.size()} chunks. Starting embedding...`);

  const items = vectorStore.items;
  const BATCH_SIZE = 25; // reduced for better reliability
  const MODEL = 'gemini-embedding-001';

  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    if (token.isCancellationRequested) break;

    const batch = items.slice(i, i + BATCH_SIZE);
    const texts = batch.map(b => b.text);

    let embeddings: number[][] | null = null;
    let attempt = 0;
    const maxAttempts = 5;

    while (attempt < maxAttempts) {
      try {
        const client = await initGenAI(context);
        output.appendLine(`Embedding batch ${i / BATCH_SIZE + 1} of ${Math.ceil(items.length / BATCH_SIZE)}...`);
        
        const resp: any = await (client as any).models.embedContent({
          model: MODEL,
          contents: texts
        });

        const extracted = Array.isArray(resp.embeddings) ? resp.embeddings : resp.data?.map((d: any) => d.embedding?.values || d.embedding);
        if (!extracted || !Array.isArray(extracted[0])) throw new Error('Invalid embedding format');
        
        embeddings = extracted as number[][];
        break;

      } catch (err: any) {
        attempt++;
        const statusCode = err?.status || err?.code || 'unknown';
        output.appendLine(`Error embedding batch ${i / BATCH_SIZE + 1}: ${getErrorMessage(err)} (status: ${statusCode}), attempt ${attempt}/${maxAttempts}`);

        if (attempt >= maxAttempts) {
          output.appendLine(`Giving up on batch ${i / BATCH_SIZE + 1}.`);
          break;
        }

        const waitMs = Math.pow(2, attempt) * 500 + Math.random() * 300; // backoff + jitter
        output.appendLine(`Retrying in ${waitMs.toFixed(0)}ms...`);
        await new Promise(res => setTimeout(res, waitMs));
      }
    }

    if (embeddings) {
      batch.forEach((b, idx) => {
        b.embedding = embeddings![idx] || null;
      });
    }

    progress.report({ message: `Embedding chunks ${Math.min(i + BATCH_SIZE, items.length)}/${items.length}`, increment: 50 / Math.ceil(items.length / BATCH_SIZE) });

    // Small delay to avoid hitting rate limits
    await new Promise(res => setTimeout(res, 200));
  }

  output.appendLine('Indexing complete.');
  output.show(true);
}

// ---------------------------
// Commands & activation
// ---------------------------

export async function activate(context: vscode.ExtensionContext) {
  console.log('semantic-code-search: activated');

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
  console.log('semantic-code-search: deactivated');
}

// small helpers
function truncate(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

/*
  NOTES:
  - Ensure your package.json contributes the commands and activationEvents:
    "activationEvents": [
      "onCommand:semanticSearch.indexWorkspace",
      "onCommand:semanticSearch.query",
      "onCommand:semanticSearch.clearIndex",
      "onCommand:semanticSearch.showStats",
      "onCommand:semanticSearch.setApiKey"
    ],
    and include corresponding contributes.commands entries.

  - Install SDK:
      npm install @google/genai

  - For local dev you can also set:
      export GOOGLE_API_KEY="..."   (or on Windows: setx GOOGLE_API_KEY "...")
    or run the Set API Key command inside the Extension Development Host.

  - Tweak BATCH_SIZE and embedding config according to rate limits and costs.
*/
