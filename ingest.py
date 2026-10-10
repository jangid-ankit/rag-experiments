import re, sys, chromadb, ollama

EMBED = "qwen3-embedding:4b"
col = chromadb.PersistentClient(path="db").get_or_create_collection(
    "fountainhead", metadata={"hnsw:space": "cosine"})

text = open(sys.argv[1], encoding="utf-8").read()
paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

# ~1200-char chunks from whole paragraphs, 200-char overlap
chunks, cur = [], ""
for p in paras:
    if cur and len(cur) + len(p) > 1200:
        chunks.append(cur.strip())
        cur = cur[-200:]
    cur += "\n\n" + p
if cur.strip():
    chunks.append(cur.strip())

B = 64
for i in range(0, len(chunks), B):
    batch = chunks[i:i + B]
    embs = ollama.embed(model=EMBED,
        input=[f"title: none | text: {c}" for c in batch]).embeddings
    col.add(ids=[str(i + j) for j in range(len(batch))],
            documents=batch, embeddings=embs,
            metadatas=[{"pos": i + j} for j in range(len(batch))])
    print(f"{i + len(batch)}/{len(chunks)} chunks embedded")