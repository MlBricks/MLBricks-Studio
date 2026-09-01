# MLB Studio Security Model

## Projects and executable components

MLB Studio project files can contain API bindings, Python imports, User Functions, and User Classes. Treat a project obtained from another person or repository as executable code.

Projects loaded from files, Hub, or cloud bundles are untrusted for the current Python session. Studio blocks executable API components and external imports until the project has been reviewed and the user explicitly runs:

```python
builder.trust_project()
```

Trust is intentionally not serialized. Reopening or downloading the project requires a new trust decision.

## Model checkpoints

Current MLBricks model artifacts should be preferred. Legacy `.pt`, `.pth`, and `.ckpt` files are loaded with PyTorch's restricted `weights_only=True` path by default. A legacy checkpoint that requires arbitrary pickle deserialization is rejected unless unsafe legacy loading is explicitly enabled for a file the user fully trusts.

## Model serving

The built-in server is safe-by-default for notebook/local use: loopback bind, same-origin CORS, API-key authentication, browser security headers, request/prompt/token limits, rate limiting, concurrency limits, and a generation timeout. Public ngrok tunnels require API-key authentication.

For a permanent public deployment, use a production reverse proxy/TLS termination layer and normal infrastructure controls such as process supervision, network policy, observability, and upstream rate limiting.

## Cloud bundles

ZIP bundle extraction rejects path traversal and symbolic-link entries.

## Reporting a vulnerability

Use the repository's GitHub Security Advisory / private vulnerability reporting channel when available. Avoid publishing exploit details in a public issue before a fix is available.
