# AI Repo Wiki

AI-powered WIKI generator for Git repositories. Clone a repo, index its files, and automatically build a structured wiki with pages and navigation—all driven by LLM analysis.

## What it does

1. **Fetch & clone** — Add a repository by URL; files are scanned and stored.
2. **Index** — LLM analyzes each file (responsibility, key elements, dependencies) and stores metadata.
3. **Subsystems** — Hierarchical clustering groups related files into subsystems.
4. **Wiki** — Generates markdown wiki pages per subsystem and a sidebar navigation tree.
5. **Export** — Push the wiki to a `wiki/` folder on branch `ai-repo-wiki` in the repo.

The app exposes a web UI to browse repos, trigger builds, and view the generated wiki. An Admin console lets you run indexing, subsystem builds, and wiki generation, plus export markdown to the repository.

## Getting started locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- [OpenAI API key](https://platform.openai.com/api-keys) (or Gemini key if using LiteLLM)

### 1. Clone and set up environment

```bash
git clone <this-repo-url>
cd ai-wiki-repo
```

Copy the example env file and add your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 2. Backend (Python)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend (Node)

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Run the app

```bash
python app.py
```

The server starts at [http://localhost:5000](http://localhost:5000).

Or use the start script (assumes `.venv` exists and frontend is built):

```bash
./scripts/start-local.sh
```

### 5. Quick workflow

1. Open the app and add a repo (e.g. `https://github.com/owner/repo.git`).
2. Go to **Admin** → **Index** and click **Build Index**.
3. After indexing completes, **Build Subsystems**.
4. Then **Build Wiki**.
5. Use **Export Markdown to Repo** to write the wiki to the repo on branch `ai-repo-wiki`.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key for LLM calls |
| `GEMINI_API_KEY` | If using Gemini | For `LLM_MODEL=gemini/...` |
| `LLM_MODEL` | No | Model name (default: `gpt-5-mini`) |
| `DATA_DIR` | No | Base dir for repos and DB (default: `./data`) |
| `DATABASE_URL` | No | SQLite path (default: `sqlite:///<DATA_DIR>/repos.db`) |

See [.env.example](.env.example) for more options.

## Project structure

```
ai-wiki-repo/
├── app.py                 # Flask app, routes, serves frontend
├── constants.py           # App constants (overridable via env)
├── repo_analyzer/         # Core logic
│   ├── db.py              # SQLite adapter
│   ├── db_managers/       # Repo, File, Subsystem, Wiki managers
│   ├── models/            # SQLAlchemy models
│   ├── services/          # Repo, File, Subsystem, Wiki services
│   └── utils/             # Async OpenAI batch runner
├── frontend/              # React + Vite + Tailwind
└── scripts/start-local.sh # Local run script
```

## Demo

Live demo: [https://ai-repo-wiki.fly.dev/](https://ai-repo-wiki.fly.dev/)
