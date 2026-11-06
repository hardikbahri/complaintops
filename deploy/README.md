# Putting ComplaintOps online

There are two ways, depending on whether you want the demo to run live.

|  | A. Static copy | B. Live copy |
|---|---|---|
| What visitors get | The whole page: results, gate, drift, speed, lessons. The demo **replays four real recorded runs** of the agent and says so. | The same page with a **live agent** on their own complaint. |
| What it needs | Any static file host. The bundle is five files (about 120 KB). | A host that runs a Docker container (about 310 MB image, about 600 MB of memory). |
| Cost | Free on common hosts. | Hugging Face now charges for Docker Spaces (PRO plan); other hosts have limits you should check. |
| Build it | `python deploy/build_static.py` | `python deploy/build_space.py` |

## A. Static copy (free)

The page detects that no API sits next to it, switches to "recorded runs" mode, makes the text box read-only, and shows a
note saying so. The recorded runs carry no trace links and no keys; `build_static.py` refuses to finish if either appears.

1. Build it: `python deploy/build_static.py` (writes `build/static`).
2. On a Hugging Face account, create a Space with SDK **Static** (the docs list only Gradio and Docker Spaces as needing
   a paid plan; if the form shows "Paid" next to Static, use another static host: the folder works anywhere).
   Create an access token with **write** permission (Settings, Access Tokens) and keep it to yourself.
3. Push it (the username is your Hugging Face username; the password is the token):
   ```
   cd build/static
   git init -b main
   git add .
   git commit -m "ComplaintOps"
   git remote add space https://huggingface.co/spaces/<your-username>/complaintops
   git push space main
   ```
4. The Space is live within a minute at `https://<your-username>-complaintops.hf.space`.

## B. Live copy (needs a Docker host)

The hosted container runs the same API and page, with no PyTorch and no database: the similar-complaint search uses an
in-memory index exported from the pgvector table, and query embeddings come from fastembed (an ONNX runtime), which
matches the stored vectors (cosine 1.0 in a check).

```
python -m complaintops.search_index export     # needs the pgvector database running; writes data/search_index/
python deploy/build_space.py                   # writes build/space (about 110 MB); refuses to finish if a key is in it
docker build -t complaintops-space build/space
docker run --rm -p 7860:7860 complaintops-space   # then open http://127.0.0.1:7860
```

To publish it on a Hugging Face Docker Space (paid) push `build/space` the same way as above, with Git LFS for the model
and the vectors (`git lfs install` first). In the Space settings add the secret `GROQ_API_KEY` and the variable
`LLM_PROVIDER=groq` for real LLM notes (without them the mock LLM is used and the page says so), optionally lower
`RATE_PER_DAY` (default 300), and **do not** set the `LANGFUSE_*` variables, or every visitor's text would be sent to a
tracing service. Any other host that runs a container can use the same image; check its memory limit (the container uses
about 600 MB) and its free-tier terms first.

## What is in each bundle

- Static: `index.html`, `app.js`, `styles.css`, `showcase.json`, `README.md`.
- Live: `src/complaintops` (only the runtime modules), `web/`, `data/onnx` (the int8 model, tokenizer, classes, offsets)
  and `data/search_index` (float16 vectors and snippets for 50,000 training complaints). No training code, no `.env`.

Rebuilding a bundle deletes and recreates its folder, including any git repository you initialised inside it, so keep a
separate clone of the Space and copy the new bundle into it when you update.
