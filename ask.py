import chromadb, ollama

EMBED, LLM, K = "qwen3-embedding:4b", "gemma4:4b", 6
col = chromadb.PersistentClient(path="db").get_collection("fountainhead")

while True:
    q = input("\nAsk: ").strip()
    if not q:
        break
    qe = ollama.embed(model=EMBED,
        input=f"task: search result | query: {q}").embeddings[0]
    docs = col.query(query_embeddings=[qe], n_results=K)["documents"][0]
    ctx = "\n\n---\n\n".join(docs)
    prompt = ("Answer using only these excerpts from The Fountainhead. "
              "If they don't contain the answer, say so.\n\n"
              f"Excerpts:\n{ctx}\n\nQuestion: {q}")
    for part in ollama.chat(model=LLM, stream=True, options={"num_ctx": 8192},
                            messages=[{"role": "user", "content": prompt}]):
        print(part["message"]["content"], end="", flush=True)
    print()