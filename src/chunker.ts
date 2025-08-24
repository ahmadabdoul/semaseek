import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript/typescript';
import JavaScript from 'tree-sitter-javascript';

const parser = new Parser();

// Simple text-based chunking for fallback
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

export function chunkCode(code: string, languageId: string) {
    switch (languageId) {
        case 'typescript':
        case 'typescriptreact':
            parser.setLanguage(TypeScript);
            break;
        case 'javascript':
        case 'javascriptreact':
            parser.setLanguage(JavaScript);
            break;
        default:
            return chunkText(code);
    }

    const tree = parser.parse(code);
    const chunks: { start: number; end: number; text: string }[] = [];
    const nodesToCapture = [
        'function_declaration',
        'method_definition',
        'class_declaration',
        'arrow_function',
        'export_statement'
    ];

    function walk(node: Parser.SyntaxNode) {
        if (nodesToCapture.includes(node.type)) {
            chunks.push({
                start: node.startIndex,
                end: node.endIndex,
                text: node.text,
            });
            // Don't traverse into children of captured nodes
            return;
        }
        node.children.forEach(walk);
    }

    walk(tree.rootNode);

    // If no specific nodes were captured, fall back to simple chunking
    if (chunks.length === 0) {
        return chunkText(code);
    }

    return chunks;
}
