import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";
import { ChatOpenAI } from "@langchain/openai";
import { program } from "commander";
import * as dotenv from "dotenv";

dotenv.config();

const CHROMA_DB_URL = process.env.CHROMA_DB_URL || "http://localhost:8000";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
	console.error("⚠️  FALTA API KEY: Debes configurar OPENAI_API_KEY en tu .env");
	process.exit(1);
}

// 📌 Configurar CLI con Commander
program
	.version("1.0.0")
	.description(
		"CLI para generar código en Golang y responder preguntas técnicas usando un sistema RAG mejorado con doble LLM",
	)
	.argument("<query>", "Consulta en lenguaje natural")
	.action(async (query) => {
		console.log(`🔎 Query original del usuario: "${query}"`);

		try {
			// 1️⃣ LLM 1 Reformula la query
			const refinedQuery = await reformulateQuery(query);
			console.log(`📝 Query optimizada para búsqueda: "${refinedQuery}"`);

			// 2️⃣ Búsqueda en ChromaDB con la query reformulada
			const context = await fetchRelevantDocs(refinedQuery);
			console.log("✅ Contexto encontrado en ChromaDB.");

			// 3️⃣ LLM 2 Responde la pregunta o genera código basado en el contexto
			const response = await answerOrGenerateCode(query, context);
			console.log("\n💻 Respuesta generada:\n");
			console.log(response);
		} catch (error) {
			console.error("❌ Error:", error);
		}
	});

program.parse(process.argv);

// 📌 1️⃣ LLM 1: Reformula la query del usuario para optimizar la búsqueda
async function reformulateQuery(query) {
	console.log("🤖 LLM 1 optimizando la consulta...");

	const llm = new ChatOpenAI({
		modelName: "gpt-4o",
		openAIApiKey: OPENAI_API_KEY,
	});

	const prompt = `Eres un asistente experto en Golang con habilidades avanzadas en recuperación de información.
Tu tarea es reformular la siguiente consulta de usuario en una búsqueda técnica que optimice la recuperación de información en la documentación de Go.

Ejemplo:
Usuario: "Cómo manejar archivos en Go?"
Búsqueda optimizada: "API para manipulación de archivos en la librería estándar de Go, incluyendo os.Open, os.ReadFile y bufio.Scanner."

Usuario: "${query}"
Búsqueda optimizada:`;

	const response = await llm.invoke(prompt);
	return response.content; // Accede al contenido del mensaje
}

// 📌 2️⃣ Buscar información en ChromaDB con la query optimizada
async function fetchRelevantDocs(query) {
	console.log("📄 Consultando ChromaDB...");

	const client = new ChromaClient({ host: CHROMA_DB_URL });
	const embeddingFunction = new OpenAIEmbeddingFunction({
		openai_api_key: OPENAI_API_KEY,
		openai_model: "text-embedding-3-small",
	});

	const collection = await client.getCollection({
		name: "golang_docs",
		embeddingFunction: embeddingFunction,
	});

	// 🔥 Generar embedding de la query
	const queryEmbedding = await embeddingFunction.generate([query]);

	// 🔎 Buscar información relevante en ChromaDB con embeddings
	const results = await collection.query({
		queryEmbeddings: queryEmbedding, // ✅ Usa embeddings explícitamente
		nResults: 5,
	});

	console.log("🔍 Resultados obtenidos:", results);
	return (
		results.documents?.[0]?.join("\n\n") ??
		"No se encontraron documentos relevantes."
	);
}

// 📌 3️⃣ LLM 2: Responde preguntas técnicas o genera código basado en el contexto
async function answerOrGenerateCode(query, context) {
	console.log("🤖 LLM 2 procesando la consulta con el contexto recuperado...");

	const llm = new ChatOpenAI({
		modelName: "gpt-4o",
		openAIApiKey: OPENAI_API_KEY,
	});

	const prompt = `Eres un experto en Golang con un profundo conocimiento del lenguaje y su ecosistema.
Tienes acceso a documentación oficial de la librería estándar de Go y puedes utilizarla para responder preguntas técnicas con información precisa.

**Reglas de respuesta:**
- Dale prioridad al uso de la documentación proporcionada.
- Si en la documentación proporcionada existe la funcionalidad de la pregunta, no dirás que no existe.
- Si y solo si la documentación no cubre la pregunta, responde usando tu conocimiento general de Go.
- Puedes usar tu base de conocimientos para complementar la respuesta.
- Responde de manera clara y concisa.
- Si la consulta requiere código, genera un ejemplo funcional y bien estructurado.
- Si es necesario, explica el código generado brevemente, pero sin ser redundante.
- No pidas al usuario que importe paquetes manualmente; inclúyelos en el código cuando sea necesario.

---

📌 **DOCUMENTACIÓN DISPONIBLE**
(Si la documentación es relevante, úsala en la respuesta)

${context}

---

**Pregunta del usuario:**
${query}

📌 **Respuesta técnica o código en Go:**`;

	const response = await llm.invoke(prompt);
	return response.content; // Accede al contenido del mensaje
}
