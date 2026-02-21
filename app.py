from pathlib import Path

from dotenv import load_dotenv

from flask import Flask, request, jsonify

load_dotenv()

from constants import DATA_DIR
from db import get_default_adapter
from repo_analyzer import clone_repo,Repo, RepoManager

app = Flask(__name__)
db_adapter = get_default_adapter()
db_adapter.create_tables()
db_adapter.migrate_tables()


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/fetch-repo", methods=["POST"])
def fetch_repo():
    """
    Clone a Git repository into data/repos/<slug>.

    JSON body: { "url": "https://github.com/owner/repo.git", "data_dir": null }
    data_dir is optional; when omitted uses default (repo root/data).
    """
    body = request.get_json(silent=True) or {}
    repo_url = body.get("url") or request.args.get("url")
    if not repo_url or not repo_url.strip():
        return jsonify({"error": "Missing 'url' in JSON body or query"}), 400

    repo_url = repo_url.strip()
 

    try:
        owner, repo_name = Repo.parse_owner_repo(repo_url)
        h = Repo.compute_hash(owner, repo_name)

        # clone_repo returns immediately (with a warning) if the folder exists
        path = clone_repo(repo_url, data_dir=DATA_DIR, target_name=h)

        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            existing = repo_manager.get_by_owner_repo(owner, repo_name)
            if existing:
                files = repo_manager.list_repo_files(existing.repo_hash, filter_scan_excluded=True)
            else:
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

        return jsonify({
            "repo_hash": existing.repo_hash,
            "files": file_list,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/repos/<string:repo_hash_>/list-files", methods=["GET"])
def list_repo_files(repo_hash_: str) -> tuple:
    """
    Return files for a repo, with optional filters.

    Query params:
        project_only: "true" — return only files where is_project_file=True.
                      Omit to return all scannable files.
    """
    project_only = request.args.get("project_only", "").lower() == "true"

    try:
        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash_)
            if repo is None:
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
                }
                for f in files
            ]

        return jsonify({
            "repo_hash": repo_hash_,
            "total": len(file_list),
            "files": file_list,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/data", methods=["DELETE"])
def clear_all_data():
    """Clear all data in the SQL DB (repo_files and repos)."""
    try:
        with db_adapter.session() as session:
            repo_manager = RepoManager(session)
            files_deleted, repos_deleted = repo_manager.clear_all()
        return jsonify({
            "repos_deleted": repos_deleted,
            "files_deleted": files_deleted,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
