import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs-extra";
import path from "path";
import * as dotenv from "dotenv";

dotenv.config();

const CHROMA_DB_PATH = process.env.CHROMA_DB_PATH || "./chroma_db";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DOCS_DIR = "./docs"; // Directory containing Markdown files

async function processMarkdownFiles(directory) {
	const documents = [];

	const files = await fs.readdir(directory);
	for (const file of files) {
		if (file.endsWith(".md")) {
			const packageName = file.replace(".md", "");
			console.log(`📄 Procesando documento: ${packageName}`);

			const content = await fs.readFile(path.join(directory, file), "utf-8");

			const textSplitter = new RecursiveCharacterTextSplitter({
				chunkSize: 2000,
				chunkOverlap: 200,
				separators: ["\n\n", "\n", " "],
			});

			const chunks = await textSplitter.createDocuments([content]);

			for (const [index, chunk] of chunks.entries()) {
				documents.push({
					pageContent: chunk.pageContent,
					metadata: {
						package: packageName,
						chunk: index,
					},
				});
			}
		}
	}
	return documents;
}

async function storeDocsInChromaDB() {
	console.log("📄 Procesando documentación Markdown...");
	const documents = await processMarkdownFiles(DOCS_DIR);

	console.log("📥 Conectando a ChromaDB en:", CHROMA_DB_PATH);
	const client = new ChromaClient({ host: "http://localhost:8000" });

	const embeddingFunction = new OpenAIEmbeddingFunction({
		openai_api_key: OPENAI_API_KEY,
		openai_model: "text-embedding-3-small",
	});
	const collection = await client.getOrCreateCollection({
		name: "golang_docs",
		embeddingFunction: embeddingFunction,
	});

	console.log("📥 Guardando documentos en ChromaDB...");
	for (const doc of documents) {
		await collection.add({
			ids: [`${doc.metadata.package}_${doc.metadata.chunk}`],
			documents: [doc.pageContent],
			metadatas: [doc.metadata],
		});
	}

	console.log(
		"✅ Todos los documentos almacenados en ChromaDB con embeddings de OpenAI.",
	);
}

storeDocsInChromaDB().catch(console.error);
