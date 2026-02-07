# Agentic Chaser

LOA (Letter of Authority) and document-chasing application for advisors. It prioritises workflows, tracks client and provider communications, and automates follow-ups.

---

## Solution overview

Agentic Chaser helps advisors manage Letters of Authority and client documents through a single dashboard and a set of agents that decide next actions, generate chase messages, and validate uploaded documents.

### Main capabilities

- **Dashboard (Streamlit)**  
  Priority queue, fact-find chasing, client/LOA/provider views, document upload with OCR validation (Tesseract), and chase message generation.

- **Workflow orchestration**  
  LangGraph-based flow that routes to client communications, provider communications, provider RPA, or document verification. The graph is defined in `orchestration/state_graph.py`.

- **Agents**  
  - Workflow orchestrator – decides next action per LOA  
  - Client communication – generates client-facing chase messages  
  - Provider communication – provider follow-up  
  - Provider RPA – provider submission actions  
  - Document processing – OCR (Tesseract) and quality checks; accepted documents are stored in client-wise folders  
  - Predictive intelligence – priority and sentiment  
  - Fact-find chasing – document requests and queue

- **Data**  
  - **PostgreSQL:** Clients, LOAs, document submissions, post-advice items, communication logs  
  - **Redis (optional):** Cache  
  - **Files:** Uploads in `data/uploads/`; accepted documents in `data/documents/{client_id}/` with clear filenames (e.g. `Passport_20250207_abc12.pdf`)

### High-level flow

1. Run the dashboard and open the Streamlit app.  
2. View priority queue, fact-find queue, and post-advice queue.  
3. Select a client or LOA and run a chase (client comms, provider comms, or RPA) or upload documents.  
4. Uploaded documents are OCR-validated; when validation passes, the file is copied to `data/documents/{client_id}/`.  
5. Optional CLI: `init-db` (create tables), `seed` (load test data), `train` (train ML models), `chaser` (run one autonomic chaser cycle).

---

## Tech stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10+ (type hints, modern syntax) |
| **UI** | Streamlit |
| **Agents / LLM** | LangGraph, langchain-core, langchain-ollama (Ollama) |
| **Database** | SQLAlchemy, PostgreSQL |
| **Cache (optional)** | Redis |
| **OCR** | pytesseract, Pillow; optional pdf2image + Poppler for PDFs |
| **Analytics / charts** | pandas, Altair |
| **ML models** | Scikit-learn–style (sentiment, priority) in `models/ml_models.py`; artifacts in `data/trained_models/` |
| **Config** | python-dotenv; settings in `config/config.py` |

### Python dependencies (from `requirements.txt`)

- altair  
- langgraph  
- langchain-core  
- langchain-ollama  
- langchain-text-splitters  
- streamlit  
- pandas  
- pytesseract  
- Pillow  

**Optional:** `pdf2image` (and system Poppler) for OCR on PDFs; without it, only image files are supported for OCR.

**System dependency:** The [Tesseract](https://github.com/tesseract-ocr/tesseract) engine must be installed and on your system PATH for document OCR. Windows: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

---

## Setup instructions

### Prerequisites

- **Required:** Python 3.10+, pip, PostgreSQL (running and reachable).  
- **Optional:** Redis, [Ollama](https://ollama.ai) (for local LLM), Tesseract (for document OCR).

### Steps

1. **Clone and enter the repo**  
   `git clone <repo-url> && cd agentic-chaser`

2. **Create and activate a virtual environment**  
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract (for document OCR)**  
   - Install from [Tesseract](https://github.com/tesseract-ocr/tesseract) or [UB-Mannheim (Windows)](https://github.com/UB-Mannheim/tesseract/wiki).  
   - Add the Tesseract install directory to your system PATH.  
   - Optional: `pip install pdf2image` and install Poppler for PDF OCR.

5. **Database**  
   Create a PostgreSQL database (e.g. `agentic_chaser`) and a user. Note host, port, user, password, and database name for `DATABASE_URL`.

6. **Environment file**  
   Create a `.env` file at the **project root** (same folder as `main.py`). The app loads it via `config/config.py`. Do not commit real credentials. Set at least `DATABASE_URL`; see “Environment variables” below.

7. **Optional**  
   Install and run [Ollama](https://ollama.ai) for local LLM; run Redis if you use cache features.

---

## Environment variables

The application loads `.env` from the **project root**. Missing optional variables can be omitted for local runs.

| Variable | Purpose | Default / example |
|----------|---------|-------------------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@127.0.0.1:5433/agentic_chaser` |
| `OLLAMA_MODEL` | Ollama model name for agents | `llama3.2:1b` |
| `OLLAMA_BASE_URL` | Ollama API base URL | `http://localhost:11434` |
| `REDIS_URL` | Redis connection (optional) | `redis://localhost:6379/0` |
| `APP_ENV` | Environment name | `development` |
| `APP_DEBUG` | Enable SQL echo / debug | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SENDGRID_API_KEY` | SendGrid API key (optional) | - |
| `FROM_EMAIL` | Sender email address | `advisor@example.com` |
| `TWILIO_ACCOUNT_SID` | Twilio account SID (optional) | - |
| `TWILIO_AUTH_TOKEN` | Twilio auth token (optional) | - |
| `TWILIO_PHONE_NUMBER` | Twilio phone number (optional) | - |

---

## Step-by-step guide to run locally

1. Install **Python 3.10+**, **PostgreSQL**, and (for OCR) **Tesseract**. Ensure they are on PATH where required.

2. Clone the repository and change into the project directory:
   ```bash
   git clone <repo-url>
   cd agentic-chaser
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS / Linux
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a PostgreSQL database (e.g. `agentic_chaser`) and a user with access.

6. Create a `.env` file at the project root. Set at least:
   ```env
   DATABASE_URL=postgresql://user:password@host:port/agentic_chaser
   ```
   Optionally set `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and other variables from the table above.

7. From the project root, create database tables:
   ```bash
   python main.py init-db
   ```

8. **(Optional)** Load test data from `data/test`:
   ```bash
   python main.py seed
   ```

9. **(Optional)** Train ML models using `data/synthetic_data`:
   ```bash
   python main.py train
   ```

10. Start the dashboard:
    ```bash
    python main.py dashboard
    ```
    Or run the default command:
    ```bash
    python main.py
    ```
    Open the URL shown in the terminal (default: http://localhost:8501).

11. **(Optional)** Run one autonomic chaser cycle:
    ```bash
    python main.py chaser
    ```

### Troubleshooting

- **OCR skipped or “OCR engine not available”**  
  Install Tesseract and add it to your system PATH. On Windows, install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

- **Database connection failed**  
  Check `DATABASE_URL` in `.env` (host, port, user, password, database name). Ensure PostgreSQL is running and the database exists.

- **Ollama model not found (404)**  
  If using Ollama, pull the model: `ollama pull <model>` (e.g. `ollama pull llama3.2:1b`). Set `OLLAMA_MODEL` in `.env` to a model from `ollama list`.

---

## Deploy on Render

You can host the dashboard and database on [Render](https://render.com) using the included Blueprint.

### One-click from Blueprint (recommended)

1. Sign in at [dashboard.render.com](https://dashboard.render.com) and connect your **GitHub** account.
2. Click **New** → **Blueprint**.
3. Connect the **agentic-chaser** repo and confirm the `render.yaml` in the root.
4. Render will create:
   - A **PostgreSQL** database (`agentic-chaser-db`, free tier).
   - A **Web Service** (`agentic-chaser`) that runs the Streamlit app.
5. The app will get `DATABASE_URL` from the linked database. On each deploy, `python main.py init-db` runs before the app starts (so tables exist).
6. After the first deploy, open the service URL (e.g. `https://agentic-chaser-xxxx.onrender.com`).

### Manual setup (without Blueprint)

1. In Render, create a **PostgreSQL** database (free tier). Note the **Internal Database URL** (or External if you need it).
2. Create a **Web Service**. Connect the same GitHub repo, branch `main`.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `streamlit run dashboard/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless true`
5. **Pre-deploy command (optional but recommended):** `python main.py init-db`
6. **Environment:** Add `DATABASE_URL` and set it to the Postgres connection string from step 1.
7. Deploy. The app will be available at the service’s `.onrender.com` URL.

### Notes for Render

- **Free tier:** The web service may spin down after inactivity; the first request can be slow to wake.
- **OCR:** The free Python environment does not include Tesseract. Document OCR will show as “OCR engine not available” unless you switch to a [Docker-based deploy](https://render.com/docs/docker) with a custom image that installs Tesseract.
- **Ollama:** Not available on Render. Leave `OLLAMA_*` unset or point to an external LLM API if you add one later.
