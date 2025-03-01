import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";
import OpenAI from "openai";
import * as dotenv from "dotenv";
import readline from "node:readline";

dotenv.config();

const CHROMA_DB_URL = process.env.CHROMA_DB_URL || "http://localhost:8000";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
	console.error("⚠️ FALTA API KEY: Debes configurar OPENAI_API_KEY en tu .env");
	process.exit(1);
}

// Inicializar el cliente de OpenAI
const openai = new OpenAI({
	apiKey: OPENAI_API_KEY,
});

// Historial de la conversación
const conversationHistory = [];

// Función para añadir mensajes al historial
function addToHistory(role, message) {
	conversationHistory.push({ role, message });
}

// Función para formatear el historial como texto
function formatHistory() {
	return conversationHistory
		.map(
			(item) =>
				`${item.role === "user" ? "Usuario" : "Asistente"}: ${item.message}`,
		)
		.join("\n");
}

// 📌 1️⃣ Reformular la consulta del usuario usando el SDK de OpenAI
async function reformulateQuery(query) {
	console.log("🤖 Optimizando la consulta...");

	const prompt = `Eres un asistente experto en Golang con habilidades avanzadas en recuperación de información.
Tu tarea es reformular la siguiente consulta de usuario en una búsqueda técnica que optimice la recuperación de información en la documentación de Go.

Ejemplo:
Usuario: "Cómo manejar archivos en Go?"
Búsqueda optimizada: "API para manipulación de archivos en la librería estándar de Go, incluyendo os.Open, os.ReadFile y bufio.Scanner."

Usuario: "${query}"
Búsqueda optimizada:`;

	const response = await openai.chat.completions.create({
		model: "gpt-4o",
		messages: [{ role: "user", content: prompt }],
		temperature: 0.7,
	});

	return response.choices[0].message.content;
}

// 📌 2️⃣ Buscar información en ChromaDB con la consulta optimizada
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

	// Generar embedding de la consulta
	const queryEmbedding = await embeddingFunction.generate([query]);

	// Buscar información relevante en ChromaDB
	const results = await collection.query({
		queryEmbeddings: queryEmbedding,
		nResults: 5,
	});

	return (
		results.documents?.[0]?.join("\n\n") ??
		"No se encontraron documentos relevantes."
	);
}

// 📌 3️⃣ Responder o generar código usando el SDK de OpenAI
async function answerOrGenerateCode(query, context) {
	console.log("🤖 Procesando la consulta con el contexto recuperado...");

	const conversationContext = formatHistory();

	const prompt = `Eres un experto en Golang con un profundo conocimiento del lenguaje y su ecosistema.
Tienes acceso a documentación oficial de la librería estándar de Go y puedes utilizarla para responder preguntas técnicas con información precisa.

**Reglas de respuesta:**
- Prioriza el uso de la documentación proporcionada.
- Si la documentación cubre la funcionalidad consultada, úsala; de lo contrario, complementa con tu conocimiento general de Go.
- Responde de forma clara y concisa.
- Si es necesario, genera ejemplos de código funcional y bien estructurado.

---

📌 **DOCUMENTACIÓN DISPONIBLE**
${context}

---

📌 **Historial de la conversación:**
${conversationContext}

---

**Pregunta del usuario:**
${query}

📌 **Respuesta técnica o código en Go:**`;

	const response = await openai.chat.completions.create({
		model: "gpt-4o",
		messages: [{ role: "user", content: prompt }],
		temperature: 0.7,
	});

	return response.choices[0].message.content;
}

// Configuración del readline para modo interactivo
const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
	prompt: "Consulta> ",
});

// Procesar cada consulta ingresada
async function processQuery(query) {
	addToHistory("user", query);

	try {
		// 1️⃣ Reformular la consulta
		const refinedQuery = await reformulateQuery(query);
		console.log(`📝 Query optimizada: "${refinedQuery}"`);

		// 2️⃣ Buscar contexto en ChromaDB
		const context = await fetchRelevantDocs(refinedQuery);
		console.log("✅ Contexto recuperado desde ChromaDB.");

		// 3️⃣ Responder o generar código
		const answer = await answerOrGenerateCode(query, context);
		console.log("\n💻 Respuesta generada:\n");
		console.log(answer);

		// Guardar la respuesta en el historial
		addToHistory("assistant", answer);
	} catch (error) {
		console.error("❌ Error al procesar la consulta:", error.message);
	}
}

// Iniciar el modo interactivo
console.log(
	"Modo interactivo iniciado. Escribe 'salir' para terminar la sesión.\n",
);
rl.prompt();

rl.on("line", async (line) => {
	const input = line.trim();
	if (input.toLowerCase() === "salir") {
		rl.close();
		return;
	}

	await processQuery(input);
	rl.prompt();
}).on("close", () => {
	console.log("Sesión terminada. ¡Hasta pronto!");
	process.exit(0);
});
