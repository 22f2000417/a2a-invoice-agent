import os
import json
import uuid
import hashlib
import threading
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from decision_engine import decide_packages

app = FastAPI(title="A2A Invoice Agent", version="1.0.0")

DB_PATH = os.getenv("DATABASE_PATH", "./data/agent.db")
BASE_URL = os.getenv("A2A_BASE_URL", "https://example.invalid/a2a/").strip()
if BASE_URL.startswith("[") and "](" in BASE_URL:
    BASE_URL = BASE_URL[1:BASE_URL.index("](")]
BASE_URL = BASE_URL.rstrip("/") + "/"

BATCH = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS = "application/vnd.ga5.invoice-action-results+json"
PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
A2A = "application/a2a+json"
ALLOWED_ACTIONS = {
    "settle_invoice", "request_approval", "hold_invoice",
    "reject_duplicate", "open_exception",
}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED"}
DB_LOCK = threading.RLock()


def connect():
    import sqlite3
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS decision_cache (
                package_hash TEXT PRIMARY KEY,
                proposal_json TEXT NOT NULL
            )
        """)


init_db()


def error(status, code):
    return JSONResponse(
        status_code=status,
        media_type=A2A,
        content={"error": {"code": code, "message": "Request rejected."}},
    )


def authenticate(request):
    expected = os.getenv("A2A_BEARER_TOKEN", "")
    header = request.headers.get("authorization", "")
    if not expected or not header.startswith("Bearer "):
        return None
    supplied = header[7:]
    if not supplied:
        return None
    if not hashlib.sha256(supplied.encode()).digest() == hashlib.sha256(expected.encode()).digest():
        return None
    return hashlib.sha256(supplied.encode()).hexdigest()


def protocol_error(request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "A2A_VERSION_REQUIRED")
    if request.method == "POST":
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != A2A:
            return error(415, "MEDIA_TYPE_REQUIRED")
    return None


def load_task(task_id, principal):
    with connect() as con:
        row = con.execute(
            "SELECT task_json FROM tasks WHERE id=? AND principal=?",
            (task_id, principal),
        ).fetchone()
    return json.loads(row["task_json"]) if row else None


def save_task(task, principal):
    with connect() as con:
        cur = con.execute(
            """UPDATE tasks SET state=?, task_json=?
               WHERE id=? AND principal=?""",
            (task["status"]["state"], canonical(task), task["id"], principal),
        )
        return cur.rowcount == 1


def part_payload(message, media_type):
    for part in message.get("parts", []):
        if isinstance(part, dict) and part.get("mediaType") == media_type:
            if isinstance(part.get("data"), dict):
                return part["data"]
    return None


def artifact(media_type, data):
    return {
        "artifactId": str(uuid.uuid4()),
        "parts": [{"mediaType": media_type, "data": data}],
    }


def cached_decisions(batch_id, policy_revision, packages):
    results, missing, hashes = {}, [], {}
    with connect() as con:
        for package in packages:
            key = fingerprint({"policyRevision": policy_revision, "package": package})
            package_id = str(package["packageId"])
            hashes[package_id] = key
            row = con.execute(
                "SELECT proposal_json FROM decision_cache WHERE package_hash=?",
                (key,),
            ).fetchone()
            if row:
                results[package_id] = json.loads(row["proposal_json"])
            else:
                missing.append(package)

    if missing:
        fresh = decide_packages(batch_id, policy_revision, missing)
        by_id = {str(p["packageId"]): p for p in fresh}
        with connect() as con:
            for package in missing:
                package_id = str(package["packageId"])
                proposal = by_id[package_id]
                con.execute(
                    "INSERT OR IGNORE INTO decision_cache(package_hash, proposal_json) VALUES (?, ?)",
                    (hashes[package_id], canonical(proposal)),
                )
                row = con.execute(
                    "SELECT proposal_json FROM decision_cache WHERE package_hash=?",
                    (hashes[package_id],),
                ).fetchone()
                results[package_id] = json.loads(row["proposal_json"])

    return [results[str(p["packageId"])] for p in packages]


def validate_initial(data):
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("batchId"), str) or not data["batchId"]:
        return False
    if not isinstance(data.get("policyRevision"), str) or not data["policyRevision"]:
        return False
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        return False
    ids = [p.get("packageId") for p in packages if isinstance(p, dict)]
    return (
        len(ids) == len(packages)
        and all(isinstance(i, str) and i for i in ids)
        and len(set(ids)) == len(ids)
    )


def make_task(message, context_id, proposals, batch_id):
    task_id = str(uuid.uuid4())
    history_message = dict(message)
    history_message.setdefault("contextId", context_id)
    history_message.setdefault("taskId", task_id)
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
        "history": [history_message],
        "artifacts": [artifact(PROPOSALS, {"batchId": batch_id, "proposals": proposals})],
        "metadata": {"createdAt": now()},
    }


def record_message(con, principal, message_id, msg_hash, task_id):
    con.execute(
        "INSERT INTO messages(principal,message_id,message_hash,task_id) VALUES(?,?,?,?)",
        (principal, message_id, msg_hash, task_id),
    )


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": "Evidence-based invoice decisions with safe action receipts.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{
            "url": BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "defaultInputModes": [BATCH, RESULTS],
        "defaultOutputModes": [PROPOSALS, RECEIPTS],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Propose safe invoice actions supported by document evidence.",
            "tags": ["invoice", "reconciliation", "approval", "safety"],
        }],
    }


@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    e = protocol_error(request)
    if e:
        return e
    principal = authenticate(request)
    if not principal:
        return error(401, "UNAUTHENTICATED")
    with connect() as con:
        rows = con.execute(
            "SELECT task_json FROM tasks WHERE principal=? ORDER BY rowid DESC",
            (principal,),
        ).fetchall()
    return JSONResponse({"tasks": [json.loads(r["task_json"]) for r in rows]}, media_type=A2A)


@app.get("/a2a/tasks/{task_id}")
async def read_task(task_id: str, request: Request):
    e = protocol_error(request)
    if e:
        return e
    principal = authenticate(request)
    if not principal:
        return error(401, "UNAUTHENTICATED")
    task = load_task(task_id, principal)
    if not task:
        return error(404, "TASK_NOT_FOUND")
    return JSONResponse(task, media_type=A2A)


@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    e = protocol_error(request)
    if e:
        return e
    principal = authenticate(request)
    if not principal:
        return error(401, "UNAUTHENTICATED")
    with DB_LOCK:
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT task_json FROM tasks WHERE id=? AND principal=?",
                (task_id, principal),
            ).fetchone()
            if not row:
                con.rollback()
                return error(404, "TASK_NOT_FOUND")
            task = json.loads(row["task_json"])
            if task["status"]["state"] in TERMINAL:
                con.rollback()
                return error(409, "TASK_TERMINAL")
            task["status"] = {"state": "TASK_STATE_CANCELED"}
            con.execute(
                "UPDATE tasks SET state=?,task_json=? WHERE id=? AND principal=?",
                (task["status"]["state"], canonical(task), task_id, principal),
            )
            con.commit()
    return JSONResponse(task, media_type=A2A)


@app.post("/a2a/message:send")
async def send_message(request: Request):
    e = protocol_error(request)
    if e:
        return e
    principal = authenticate(request)
    if not principal:
        return error(401, "UNAUTHENTICATED")
    try:
        body = await request.json()
    except Exception:
        return error(400, "INVALID_JSON")
    if not isinstance(body, dict):
        return error(400, "INVALID_BODY")
    message = body.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("messageId"), str) or not message["messageId"]:
        return error(400, "INVALID_MESSAGE")
    msg_id = message["messageId"]
    msg_hash = fingerprint(message)

    with DB_LOCK:
        with connect() as con:
            old = con.execute(
                "SELECT message_hash,task_id FROM messages WHERE principal=? AND message_id=?",
                (principal, msg_id),
            ).fetchone()
            if old:
                if old["message_hash"] != msg_hash:
                    return error(409, "IDEMPOTENCY_CONFLICT")
                row = con.execute(
                    "SELECT task_json FROM tasks WHERE id=? AND principal=?",
                    (old["task_id"], principal),
                ).fetchone()
                if row:
                    return JSONResponse({"task": json.loads(row["task_json"])}, media_type=A2A)
                return error(500, "TASK_STATE_UNAVAILABLE")

        result_data = part_payload(message, RESULTS)
        if result_data is not None:
            task_id = message.get("taskId")
            if not isinstance(task_id, str) or not task_id:
                return error(400, "TASK_ID_REQUIRED")
            with connect() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute(
                    "SELECT task_json FROM tasks WHERE id=? AND principal=?",
                    (task_id, principal),
                ).fetchone()
                if not row:
                    con.rollback()
                    return error(404, "TASK_NOT_FOUND")
                task = json.loads(row["task_json"])
                if task.get("contextId") != message.get("contextId"):
                    con.rollback()
                    return error(409, "CONTEXT_MISMATCH")
                if task["status"]["state"] in TERMINAL:
                    con.rollback()
                    return error(409, "TASK_TERMINAL")
                proposal_data = next(
                    (a["parts"][0]["data"] for a in task.get("artifacts", [])
                     if a.get("parts") and a["parts"][0].get("mediaType") == PROPOSALS),
                    None,
                )
                if not isinstance(proposal_data, dict) or result_data.get("batchId") != proposal_data.get("batchId"):
                    con.rollback()
                    return error(409, "BATCH_MISMATCH")
                proposals = {p["packageId"]: p for p in proposal_data.get("proposals", [])}
                results = result_data.get("results")
                if not isinstance(results, list) or len(results) != len(proposals):
                    con.rollback()
                    return error(400, "INVALID_RESULTS")
                seen, receipts = set(), []
                for r in results:
                    if not isinstance(r, dict):
                        con.rollback()
                        return error(400, "INVALID_RESULTS")
                    pid = r.get("packageId")
                    p = proposals.get(pid)
                    if not p or pid in seen or r.get("actionId") != p.get("actionId") or r.get("action") != p.get("action"):
                        con.rollback()
                        return error(409, "RESULT_MISMATCH")
                    seen.add(pid)
                    if r.get("outcome") not in ("ACCEPTED", "REJECTED"):
                        con.rollback()
                        return error(400, "INVALID_OUTCOME")
                    nonce = r.get("receiptNonce")
                    if not isinstance(nonce, str) or not nonce:
                        con.rollback()
                        return error(400, "INVALID_RECEIPT_NONCE")
                    if r["outcome"] == "ACCEPTED":
                        receipts.append({
                            "packageId": pid, "actionId": p["actionId"],
                            "action": p["action"], "receiptNonce": nonce,
                            "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
                        })
                task["history"].append(message)
                task["status"] = {"state": "TASK_STATE_COMPLETED"}
                task["artifacts"].append(artifact(RECEIPTS, {
    			"batchId": proposal_data["batchId"], "executions": receipts,
		}))
                con.execute(
                    "UPDATE tasks SET state=?,task_json=? WHERE id=? AND principal=?",
                    (task["status"]["state"], canonical(task), task_id, principal),
                )
                record_message(con, principal, msg_id, msg_hash, task_id)
                con.commit()
            return JSONResponse(task, media_type=A2A)

        data = part_payload(message, BATCH)
        if not validate_initial(data):
            return error(400, "INVALID_BATCH")
        context_id = message.get("contextId") or str(uuid.uuid4())
        try:
            proposals = cached_decisions(data["batchId"], data["policyRevision"], data["packages"])
        except Exception:
            return error(502, "DECISION_ENGINE_FAILED")
        for p in proposals:
            p["actionId"] = str(uuid.uuid4())
        task = make_task(message, context_id, proposals, data["batchId"])
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute(
                "SELECT message_hash,task_id FROM messages WHERE principal=? AND message_id=?",
                (principal, msg_id),
            ).fetchone()
            if old:
                con.rollback()
                if old["message_hash"] != msg_hash:
                    return error(409, "IDEMPOTENCY_CONFLICT")
                existing = load_task(old["task_id"], principal)
                return JSONResponse({"task": existing}, media_type=A2A) if existing else error(500, "TASK_STATE_UNAVAILABLE")
            con.execute(
                "INSERT INTO tasks(id,principal,context_id,state,task_json) VALUES(?,?,?,?,?)",
                (task["id"], principal, context_id, task["status"]["state"], canonical(task)),
            )
            record_message(con, principal, msg_id, msg_hash, task["id"])
            con.commit()
        return JSONResponse({"task": task}, media_type=A2A)
