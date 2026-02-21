import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from flask import Flask, request, jsonify
from openai import OpenAI

load_dotenv()

from constants import DATA_DIR
from db import get_default_adapter
from repo_analyzer import clone_repo, Repo, RepoManager, index_repo
from repo_analyzer.code_analyzer import FileForIndex, index_file, IndexTaskStatus
from repo_analyzer.models import IndexTask, RepoFile, RepoFileMetadata

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
db_adapter = get_default_adapter()
db_adapter.create_tables()
db_adapter.migrate_tables()


@app.route("/")
def hello_world() -> str:
    logger.debug("Health check request received.")
    return "Hello, World!"


@app.route("/fetch-repo", methods=["POST"])
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
        owner, repo_name = Repo.parse_owner_repo(repo_url)
        h = Repo.compute_hash(owner, repo_name)

        # clone_repo returns immediately (with a warning) if the folder exists
        path = clone_repo(repo_url, data_dir=DATA_DIR, target_name=h)

        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            existing = repo_manager.get_by_owner_repo(owner, repo_name)
            if existing:
                logger.info("Repo already exists: %s", existing.repo_hash)
                files = repo_manager.list_repo_files(existing.repo_hash, filter_scan_excluded=True)
            else:
                logger.info("Creating repo entry and scanning files for %s/%s", owner, repo_name)
                repo = repo_manager.add_repo_from_url(repo_url, path)
                repo_manager.populate_files_for_repo(repo.repo_hash, path)
                files = repo_manager.list_repo_files(repo.repo_hash, filter_scan_excluded=True)
                existing = repo

            file_list = [
                {
                    "file_id": f.file_id,
                    "file_path": f.file_path,
                    "file_name": f.file_name,
                    "is_project_file": f.is_project_file,
                    "is_scan_excluded": f.is_scan_excluded,
                }
                for f in files
            ]

        logger.debug("Returning %d files for repo_hash=%s", len(file_list), existing.repo_hash)
        return jsonify({
            "repo_hash": existing.repo_hash,
            "files": file_list,
        }), 200
    except Exception as e:
        logger.exception("fetch-repo failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/repos/<string:repo_hash_>/files", methods=["GET"])
@app.route("/repo/<string:repo_hash_>/files", methods=["GET"])
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
        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash_)
            if repo is None:
                logger.info("Repo not found: %s", repo_hash_)
                return jsonify({"error": f"Repo not found: {repo_hash_}"}), 404

            files = repo_manager.list_repo_files(
                repo_hash_,
                filter_scan_excluded=True,
                project_file=True if project_only else None,
            )
            file_list = [
                {
                    "file_id": f.file_id,
                    "file_path": f.file_path,
                    "file_name": f.file_name,
                    "is_project_file": f.is_project_file,
                    "is_scan_excluded": f.is_scan_excluded,
                    "metadata": f.get_metadata(),
                }
                for f in files
            ]

        logger.debug("Returning %d files for repo_hash=%s", len(file_list), repo_hash_)
        return jsonify({
            "repo_hash": repo_hash_,
            "total": len(file_list),
            "files": file_list,
        }), 200
    except Exception as e:
        logger.exception("list-files failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/repos/<string:repo_hash_>/index", methods=["POST"])
def index_repo_endpoint(repo_hash_: str) -> tuple:
    """Start or resume repo indexing and return task status."""
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


@app.route("/repos/<string:repo_hash_>/index", methods=["GET"])
def get_index_status(repo_hash_: str) -> tuple:
    """Return current indexing task status without starting a new task."""
    try:
        logger.info("Index status requested: %s", repo_hash_)
        status = _get_index_task_status(repo_hash_)
        if status is None:
            logger.info("No index task found for repo_hash=%s", repo_hash_)
            return jsonify({"error": f"No indexing task found for repo: {repo_hash_}"}), 404
        logger.debug("Index status: %s", status)
        return jsonify(status), 200
    except Exception as e:
        logger.exception("get index status failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/repos/<string:repo_hash_>/files/<int:file_id>/reindex", methods=["POST"])
def reindex_single_file(repo_hash_: str, file_id: int) -> tuple:
    """Re-index a single file in a repo and return updated metadata."""
    try:
        logger.info("Reindex file requested: repo_hash=%s file_id=%d", repo_hash_, file_id)
        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash_)
            if repo is None:
                logger.info("Repo not found: %s", repo_hash_)
                return jsonify({"error": f"Repo not found: {repo_hash_}"}), 404

            repo_file = (
                session.query(RepoFile)
                .filter(RepoFile.repo_hash == repo_hash_, RepoFile.file_id == file_id)
                .first()
            )
            if repo_file is None:
                logger.info("File not found: %d", file_id)
                return jsonify({"error": f"File not found: {file_id}"}), 404

            meta = _index_repo_file(repo, repo_file)
            logger.debug("Reindex complete for file_id=%d", file_id)
            return jsonify({
                "repo_hash": repo_hash_,
                "file_id": file_id,
                "metadata": meta,
            }), 200
    except Exception as e:
        logger.exception("reindex single file failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/repos/<string:repo_hash_>/index/stop", methods=["POST"])
def stop_indexing(repo_hash_: str) -> tuple:
    """Stop all running indexing tasks for a repo."""
    try:
        logger.info("Stop indexing requested: %s", repo_hash_)
        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash_)
            if repo is None:
                logger.info("Repo not found: %s", repo_hash_)
                return jsonify({"error": f"Repo not found: {repo_hash_}"}), 404

            tasks = (
                session.query(IndexTask)
                .filter(IndexTask.repo_hash == repo_hash_, IndexTask.status == "running")
                .all()
            )
            now = int(time.time())
            stopped = 0
            for task in tasks:
                task.status = "stopped"
                task.updated_at = now
                stopped += 1
            logger.info("Stopped %d tasks for repo_hash=%s", stopped, repo_hash_)
            return jsonify({"repo_hash": repo_hash_, "stopped_tasks": stopped}), 200
    except Exception as e:
        logger.exception("stop indexing failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/data", methods=["DELETE"])
def clear_all_data() -> tuple:
    """Clear all data in the SQL DB (repo_files and repos)."""
    try:
        logger.info("Clearing all data.")
        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            files_deleted, repos_deleted = repo_manager.clear_all()
        logger.info("Data cleared: repos=%d files=%d", repos_deleted, files_deleted)
        return jsonify({
            "repos_deleted": repos_deleted,
            "files_deleted": files_deleted,
        }), 200
    except Exception as e:
        logger.exception("clear all data failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/openai/health", methods=["GET"])
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


def _index_repo_file(repo: Repo, repo_file: RepoFile) -> RepoFileMetadata:
    rel_path = Path(repo_file.full_rel_path())
    file_path = repo.clone_path_resolved / rel_path
    logger.debug("Indexing file: %s", file_path)
    content = _read_file_text(file_path)
    summaries = index_file([
        FileForIndex(
            file_path=rel_path.as_posix(),
            file_name=repo_file.file_name,
            content=content,
        )
    ])
    summary = summaries[0]
    meta: RepoFileMetadata = {
        "responsibility": summary["responsibility"],
        "key_elements": summary["key_elements"],
        "dependent_files": summary["dependent_files"],
        "entry_point": summary["entry_point"],
    }
    repo_file.set_metadata(meta)
    repo_file.last_index_at = int(time.time())
    return meta


def _read_file_text(path: Path, *, max_bytes: int = 200_000) -> str:
    if not path.exists():
        logger.debug("File does not exist: %s", path)
        return ""
    if not path.is_file():
        logger.debug("Path is not a file: %s", path)
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        logger.debug("Truncating file %s to %d bytes", path, max_bytes)
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _get_index_task_status(repo_hash: str) -> IndexTaskStatus | None:
    with db_adapter.session() as session:
        task = (
            session.query(IndexTask)
            .filter(IndexTask.repo_hash == repo_hash)
            .order_by(IndexTask.created_at.desc())
            .first()
        )
        if task is None:
            return None
        remaining = max(0, task.total_files - task.completed_files)
        logger.debug(
            "Index task status repo_hash=%s status=%s completed=%d total=%d",
            repo_hash,
            task.status,
            task.completed_files,
            task.total_files,
        )
        return IndexTaskStatus(
            repo_hash=task.repo_hash,
            status=task.status,
            total_files=task.total_files,
            completed_files=task.completed_files,
            remaining_files=remaining,
            task_id=task.task_id,
        )


if __name__ == "__main__":
    app.run(debug=True)
