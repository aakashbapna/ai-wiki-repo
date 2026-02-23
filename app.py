import logging
from pathlib import Path

from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_from_directory
import openai._utils._logs
openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)
from openai import OpenAI
load_dotenv()
from constants import DATA_DIR
from repo_analyzer.db import get_default_adapter
from repo_analyzer import clone_repo, index_repo
from repo_analyzer.services import FileService, RepoService, SubsystemService, WikiService

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = Flask(__name__, static_folder=None)
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
db_adapter = get_default_adapter()
db_adapter.create_tables()
db_adapter.migrate_tables()


@app.route("/api/")
def hello_world() -> str:
    logger.debug("Health check request received.")
    return "Hello, World!"


@app.route("/api/fetch-repo", methods=["POST"])
def fetch_repo() -> tuple:
    """
    Clone a Git repository into data/repos/<slug>.

    JSON body: { "url": "https://github.com/owner/repo.git", "data_dir": null }
    data_dir is optional; when omitted uses default (repo root/data).
    """
    body = request.get_json(silent=True) or {}
    repo_url = body.get("url") or request.args.get("url")
    if not repo_url or not repo_url.strip():
        logger.info("fetch-repo missing url.")
        return jsonify({"error": "Missing 'url' in JSON body or query"}), 400

    repo_url = repo_url.strip()
 

    try:
        logger.info("Fetching repo: %s", repo_url)
        result = RepoService.fetch_repo_with_files(
            repo_url=repo_url,
            clone_repo_fn=clone_repo,
            data_dir=DATA_DIR,
        )
        logger.debug("Returning %d files for repo_hash=%s", len(result["files"]), result["repo_hash"])
        return jsonify(result), 200
    except Exception as e:
        logger.exception("fetch-repo failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos", methods=["GET"])
def list_repos() -> tuple:
    """Return all repos."""
    try:
        logger.info("List repos requested.")
        repos = RepoService.list_repos()
        return jsonify({"total": len(repos), "repos": repos}), 200
    except Exception as e:
        logger.exception("list repos failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>", methods=["GET"])
def get_repo_detail(repo_hash_: str) -> tuple:
    """Return a single repo by hash."""
    try:
        logger.info("Repo detail requested: %s", repo_hash_)
        repo = RepoService.get_repo_detail(repo_hash_)
        return jsonify(repo), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Repo detail validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("repo detail failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/files", methods=["GET"])
@app.route("/api/repo/<string:repo_hash_>/files", methods=["GET"])
def list_repo_files(repo_hash_: str) -> tuple:
    """
    Return files for a repo, with optional filters.

    Query params:
        project_only: "true" — return only files where is_project_file=True.
                      Omit to return all scannable files.
    """
    project_only = request.args.get("project_only", "").lower() == "true"
    logger.info("Listing files for repo_hash=%s project_only=%s", repo_hash_, project_only)

    try:
        result = FileService.list_repo_files(repo_hash_, project_only=project_only)
        logger.debug("Returning %d files for repo_hash=%s", result["total"], repo_hash_)
        return jsonify(result), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("List files validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("list-files failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/index", methods=["POST"])
def index_repo_endpoint(repo_hash_: str) -> tuple:
    """Start repo indexing and return immediately; heavy work runs in background."""
    try:
        logger.info("Index repo requested: %s", repo_hash_)
        status = index_repo(repo_hash_)
        logger.debug("Index status: %s", status)
        return jsonify(status), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Index repo validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("index repo failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/index", methods=["GET"])
def get_index_status(repo_hash_: str) -> tuple:
    """Return current indexing task status without starting a new task."""
    try:
        logger.info("Index status requested: %s", repo_hash_)
        status = FileService.get_index_task_status(repo_hash_)
        if status is None:
            logger.info("No index task found for repo_hash=%s", repo_hash_)
            return jsonify({"error": f"No indexing task found for repo: {repo_hash_}"}), 404
        logger.debug("Index status: %s", status)
        return jsonify(status), 200
    except Exception as e:
        logger.exception("get index status failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/subsystems/build", methods=["POST"])
def build_subsystems(repo_hash_: str) -> tuple:
    """Rebuild subsystems for a repo and return task status."""
    try:
        logger.info("Subsystem build requested: %s", repo_hash_)
        status = SubsystemService.build_subsystems(repo_hash_)
        logger.debug("Subsystem build status: %s", status)
        return jsonify(status), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Subsystem build validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("subsystem build failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/subsystems/build", methods=["GET"])
def get_subsystem_build_status(repo_hash_: str) -> tuple:
    """Return current subsystem build status without starting a new task."""
    try:
        logger.info("Subsystem build status requested: %s", repo_hash_)
        status = SubsystemService.get_build_status(repo_hash_)
        if status is None:
            logger.info("No subsystem task found for repo_hash=%s", repo_hash_)
            return jsonify({"error": f"No subsystem task found for repo: {repo_hash_}"}), 404
        logger.debug("Subsystem build status: %s", status)
        return jsonify(status), 200
    except Exception as e:
        logger.exception("get subsystem build status failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/subsystems", methods=["GET"])
def get_subsystems(repo_hash_: str) -> tuple:
    """Return subsystems for a repo."""
    try:
        logger.info("Subsystems requested: %s", repo_hash_)
        subsystems = SubsystemService.list_subsystems(repo_hash_)
        return jsonify({
            "repo_hash": repo_hash_,
            "total": len(subsystems),
            "subsystems": subsystems,
        }), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Get subsystems validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("get subsystems failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/wiki/build", methods=["POST"])
def build_wiki(repo_hash_: str) -> tuple:
    """Build wiki pages and sidebars for a repo."""
    try:
        logger.info("Wiki build requested: %s", repo_hash_)
        status = WikiService.build_wiki(repo_hash_)
        logger.debug("Wiki build status: %s", status)
        return jsonify(status), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Wiki build validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("wiki build failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/wiki/build", methods=["GET"])
def get_wiki_build_status(repo_hash_: str) -> tuple:
    """Return current wiki build status without starting a new task."""
    try:
        logger.info("Wiki build status requested: %s", repo_hash_)
        status = WikiService.get_build_status(repo_hash_)
        if status is None:
            logger.info("No wiki task found for repo_hash=%s", repo_hash_)
            return jsonify({"error": f"No wiki task found for repo: {repo_hash_}"}), 404
        logger.debug("Wiki build status: %s", status)
        return jsonify(status), 200
    except Exception as e:
        logger.exception("get wiki build status failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/wiki/sidebars", methods=["GET"])
def get_wiki_sidebars(repo_hash_: str) -> tuple:
    """Return wiki sidebars for a repo."""
    try:
        logger.info("Wiki sidebars requested: %s", repo_hash_)
        sidebars = WikiService.list_sidebars(repo_hash_)
        return jsonify({"repo_hash": repo_hash_, "total": len(sidebars), "sidebars": sidebars}), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Wiki sidebars validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("wiki sidebars failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/wiki/pages", methods=["GET"])
def get_wiki_pages(repo_hash_: str) -> tuple:
    """Return wiki pages for a repo."""
    try:
        logger.info("Wiki pages requested: %s", repo_hash_)
        pages = WikiService.list_pages(repo_hash_)
        return jsonify({"repo_hash": repo_hash_, "total": len(pages), "pages": pages}), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Wiki pages validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("wiki pages failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/wiki/pages/<int:page_id>", methods=["GET"])
def get_wiki_page_with_contents(repo_hash_: str, page_id: int) -> tuple:
    """Return a wiki page and its contents."""
    try:
        logger.info("Wiki page requested: %s page_id=%d", repo_hash_, page_id)
        result = WikiService.get_page_with_contents(repo_hash_, page_id)
        return jsonify(result), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") or msg.startswith("Page not found") else 400
        logger.info("Wiki page validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("wiki page failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/files/<int:file_id>/reindex", methods=["POST"])
def reindex_single_file(repo_hash_: str, file_id: int) -> tuple:
    """Re-index a single file in a repo and return updated metadata."""
    try:
        logger.info("Reindex file requested: repo_hash=%s file_id=%d", repo_hash_, file_id)
        result = FileService.reindex_single_file(repo_hash_, file_id)
        logger.debug("Reindex complete for file_id=%d", file_id)
        return jsonify(result), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") or msg.startswith("File not found") else 400
        logger.info("Reindex file validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("reindex single file failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/files/<int:file_id>/content", methods=["GET"])
def get_repo_file_content(repo_hash_: str, file_id: int) -> tuple:
    """Return raw file content for a repo file."""
    try:
        logger.info("Repo file content requested: %s file_id=%d", repo_hash_, file_id)
        result = FileService.get_repo_file_content(repo_hash_, file_id)
        return jsonify(result), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") or msg.startswith("File not found") else 400
        logger.info("Repo file content validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("get repo file content failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repos/<string:repo_hash_>/index/stop", methods=["POST"])
def stop_indexing(repo_hash_: str) -> tuple:
    """Stop all running indexing tasks for a repo."""
    try:
        logger.info("Stop indexing requested: %s", repo_hash_)
        result = FileService.stop_indexing(repo_hash_)
        logger.info("Stopped %d tasks for repo_hash=%s", result["stopped_tasks"], repo_hash_)
        return jsonify(result), 200
    except ValueError as e:
        msg = str(e)
        code = 404 if msg.startswith("Repo not found") else 400
        logger.info("Stop indexing validation failed: %s", msg)
        return jsonify({"error": msg}), code
    except Exception as e:
        logger.exception("stop indexing failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/data", methods=["DELETE"])
def clear_all_data() -> tuple:
    """Clear all data in the SQL DB (repo_files and repos)."""
    try:
        logger.info("Clearing all data.")
        result = RepoService.clear_all_data()
        logger.info("Data cleared: repos=%d files=%d", result["repos_deleted"], result["files_deleted"])
        return jsonify(result), 200
    except Exception as e:
        logger.exception("clear all data failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/openai/health", methods=["GET"])
def openai_health() -> tuple:
    """Validate OpenAI SDK configuration."""
    try:
        client = OpenAI()
        models = client.models.list()
        model_count = len(list(models))
        logger.info("OpenAI health ok. Models=%d", model_count)
        return jsonify({"status": "ok", "models": model_count}), 200
    except Exception as e:
        logger.exception("OpenAI health failed.")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str) -> object:
    if FRONTEND_DIST.exists() and path:
        if path.startswith("assets/"):
            asset_path = FRONTEND_DIST / path
            if asset_path.is_file():
                return send_from_directory(FRONTEND_DIST, path)
        file_path = FRONTEND_DIST / path
        if file_path.is_file():
            return send_from_directory(FRONTEND_DIST, path)
    if FRONTEND_DIST.exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"error": "Frontend build not found."}), 404


if __name__ == "__main__":
    app.run(debug=True)
