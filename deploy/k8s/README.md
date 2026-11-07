# Running ComplaintOps on Kubernetes (local, with kind)

- `kind` runs a one-node Kubernetes cluster inside Docker, so this needs no cloud account.
- `complaintops.yaml` is a Deployment (one pod) and a Service. `kind-cluster.yaml` publishes the Service on http://localhost:8080.
- What the Deployment sets: a startup, a readiness and a liveness probe (`/health`, `/ready`, `/health`), memory and CPU requests and limits (600 Mi requested against about 580 MB measured, 900 Mi limit), a non-root user with no extra privileges, and an optional secret for a real LLM key.
- Tracing variables are deliberately not set, so visitors' text is never sent to Langfuse.

## Try it
1. Build the image (needs `data/onnx` and `data/search_index`, see `deploy/README.md`):
   ```
   python deploy/build_space.py
   docker build -t complaintops-space build/space
   ```
2. Create the cluster, load the image and apply the manifests:
   ```
   kind create cluster --config deploy/k8s/kind-cluster.yaml
   kind load docker-image complaintops-space:latest --name complaintops
   kubectl apply -f deploy/k8s/complaintops.yaml
   kubectl rollout status deployment/complaintops
   ```
3. Open http://localhost:8080 (the page) or `curl localhost:8080/ready`.
4. Optional, a real LLM for reviewer notes: `kubectl create secret generic complaintops-llm --from-literal=GROQ_API_KEY=<your key>`, then `kubectl set env deployment/complaintops LLM_PROVIDER=groq`.
5. Clean up: `kind delete cluster --name complaintops`.

## What was checked on a real cluster
- The pod became ready in a few seconds with 0 restarts; `/health` and `/ready` returned 200.
- A confident complaint was answered by the classifier in about 170 ms; an unsure one was flagged for review with a note in about 1.2 s (mock LLM).
- Deleting the pod made Kubernetes start a replacement, which served traffic again after 24 s.

## What this is not
- It is a local demo of the manifests, not a production setup. One replica, no ingress, no autoscaling, no persistent storage (the review queue is a file inside the pod and is lost on restart).
- CI checks the manifests against the app and the Dockerfile without a cluster (`tests/test_k8s.py`). It does not build the image or start kind, because the model files are not in git.
