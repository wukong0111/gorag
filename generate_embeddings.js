import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";
import fs from "fs-extra";
import path from "path";
import * as dotenv from "dotenv";
import pLimit from "p-limit";

dotenv.config();

const CHROMA_DB_PATH = process.env.CHROMA_DB_PATH || "./chroma_db";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DOCS_DIR = process.env.DOCS_DIR || "./docs";
const CHUNK_SIZE = process.env.CHUNK_SIZE
	? Number.parseInt(process.env.CHUNK_SIZE)
	: 2000; // Volvemos a 2000
const CHUNK_OVERLAP = process.env.CHUNK_OVERLAP
	? Number.parseInt(process.env.CHUNK_OVERLAP)
	: 200;
const CONCURRENCY = 4; // Procesar 4 archivos/chunks a la vez

// Dividir texto respetando separadores, similar a RecursiveCharacterTextSplitter
function splitText(text, chunkSize = CHUNK_SIZE, chunkOverlap = CHUNK_OVERLAP) {
	const chunks = [];
	const separators = ["\n\n", "\n", " "];
	let remainingText = text;

	while (remainingText.length > 0) {
		let chunk = "";
		let foundSeparator = false;

		for (const sep of separators) {
			const nextSplit = remainingText.indexOf(sep);
			if (nextSplit !== -1 && nextSplit <= chunkSize) {
				chunk = remainingText.slice(0, nextSplit + sep.length);
				remainingText = remainingText.slice(nextSplit + sep.length);
				foundSeparator = true;
				break;
			}
		}

		if (!foundSeparator) {
			chunk = remainingText.slice(0, chunkSize);
			remainingText = remainingText.slice(chunkSize - chunkOverlap);
		}

		if (chunk.length > 0) chunks.push(chunk);
		if (remainingText.length <= chunkOverlap) {
			if (remainingText.length > 0) chunks.push(remainingText);
			break;
		}
	}
	return chunks;
}

async function processAndStoreFile(collection, filePath) {
	const packageName = path.basename(filePath, ".md");
	console.log(`📄 Procesando documento: ${packageName}`);

	const content = await fs.readFile(filePath, "utf-8");
	const chunks = splitText(content);

	const limit = pLimit(CONCURRENCY);
	await Promise.all(
		chunks.map((chunk, index) =>
			limit(() =>
				collection.add({
					ids: [`${packageName}_${index}`],
					documents: [chunk],
					metadatas: [{ package: packageName, chunk: index }],
				}),
			),
		),
	);
}

async function storeDocsInChromaDB() {
	if (!OPENAI_API_KEY) throw new Error("Falta OPENAI_API_KEY en .env");

	console.log("📥 Conectando a ChromaDB...");
	const client = new ChromaClient({ host: "http://localhost:8000" });
	const embeddingFunction = new OpenAIEmbeddingFunction({
		openai_api_key: OPENAI_API_KEY,
		openai_model: "text-embedding-3-small",
	});

	console.log("🗑️ Eliminando colección anterior 'golang_docs'...");
	try {
		await client.deleteCollection({ name: "golang_docs" });
		console.log("✅ Colección anterior eliminada.");
	} catch (error) {
		console.log("ℹ️ No había colección previa para eliminar.");
	}

	console.log("📥 Creando nueva colección 'golang_docs'...");
	const collection = await client.createCollection({
		name: "golang_docs",
		embeddingFunction,
	});

	console.log("📄 Procesando y guardando documentos en ChromaDB...");
	const files = await fs.readdir(DOCS_DIR);
	const limit = pLimit(CONCURRENCY);

	await Promise.all(
		files
			.filter((file) => file.endsWith(".md"))
			.map((file) =>
				limit(() => processAndStoreFile(collection, path.join(DOCS_DIR, file))),
			),
	);

	console.log("✅ Todos los documentos almacenados en ChromaDB.");
}

storeDocsInChromaDB().catch(console.error);
