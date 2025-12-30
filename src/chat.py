import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

from search import search_prompt

load_dotenv()
for k in ("OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")


query = "Liste o nome das empresas de telecom?"

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-large"))

llm = ChatOpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5"),
    temperature=0,
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)

results = store.similarity_search_with_score(query, k=20)

context_parts = []

for i, (doc, score) in enumerate(results, start=1):
    print("="*50)
    print(f"Resultado {i} (score: {score:.2f}):")
    print("="*50)

    print("\nTexto:\n")
    print(doc.page_content.strip())
    context_parts.append(doc.page_content.strip())

    print("\nMetadados:\n")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")
    if doc.metadata:
        context_parts.append("\n".join(f"{k}: {v}" for k, v in doc.metadata.items()))

contexto = "\n\n".join(part for part in context_parts if part).strip()

if not contexto:
    print("\nNenhum contexto encontrado para responder à pergunta.")
else:
    prompt = search_prompt(query, contexto=contexto)
    resposta = llm.invoke(prompt)
    print("\nResposta baseada no CONTEXTO:\n")
    print(resposta.content.strip())
