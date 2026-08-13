"""
Banzai Mundo — a global marketplace for human skills.
Anyone in the world posts what they know how to do; anyone else books a session.
AI-powered real-time translation removes the language barrier that has always
kept this kind of exchange local instead of global — the same barrier Uber
removed for transportation by removing the "how do I find a car nearby" barrier.

This is an MVP: real, working, deployable — not a mockup. It is intentionally
scoped smaller than Banzai (the business-OS product) since that took many
sessions of accumulated work. This is the solid foundation to build on.
"""
from __future__ import annotations

import os
import json
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

DB_PATH = os.environ.get("DB_PATH", "banzai_mundo.db")

CATEGORIES = [
    "Programación y Tecnología", "Idiomas", "Música", "Diseño y Arte",
    "Negocios y Marketing", "Educación y Tutorías", "Cocina", "Fitness y Bienestar",
    "Reparaciones y Oficios", "Escritura y Traducción", "Fotografía y Video",
    "Consultoría Legal y Fiscal", "Otro",
]

LANGUAGES = {
    "es": "Español", "en": "English", "pt": "Português", "fr": "Français",
    "de": "Deutsch", "it": "Italiano", "zh": "中文", "ja": "日本語",
}


# ── DB ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    country TEXT,
    language TEXT NOT NULL DEFAULT 'es',
    bio TEXT DEFAULT '',
    avatar_seed TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    price REAL NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD',
    session_minutes INTEGER NOT NULL DEFAULT 60,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id INTEGER NOT NULL,
    requester_id INTEGER NOT NULL,
    provider_id INTEGER NOT NULL,
    initial_message TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (skill_id) REFERENCES skills(id),
    FOREIGN KEY (requester_id) REFERENCES users(id),
    FOREIGN KEY (provider_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    sender_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    translated_text TEXT,
    translated_for_lang TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL UNIQUE,
    reviewer_id INTEGER NOT NULL,
    reviewed_user_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);
"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(get_db()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


# ── Password hashing (stdlib only, no extra dependency needed) ─────────────

import hashlib


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return secrets.compare_digest(check, digest)
    except Exception:
        return False


def is_valid_email(value: str) -> bool:
    return bool(value) and bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()))


# ── Auth helpers ─────────────────────────────────────────────────────────

def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            return jsonify({"ok": False, "error": "No autenticado"}), 401
        return f(user, *args, **kwargs)
    return wrapper


def json_error(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


# ── AI translation (the core "unlock" that makes this global) ──────────────

def translate_text(text: str, target_lang_code: str) -> str | None:
    """Translates a message to the recipient's language. Returns None if
    translation isn't available/needed, so the frontend can fall back to
    showing the original text only."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None or not text.strip():
        return None
    target_name = LANGUAGES.get(target_lang_code, target_lang_code)
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": f"Translate the user's message to {target_name}. Reply with ONLY the translation, no notes, no quotes."},
                {"role": "user", "content": text},
            ],
            max_tokens=400, temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ── Routes: pages ────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/version")
def api_version():
    return jsonify({"ok": True, "app": "Banzai Mundo", "version": "0.1.0-mvp"})


@app.get("/api/categories")
def api_categories():
    return jsonify({"ok": True, "categories": CATEGORIES, "languages": LANGUAGES})


# ── Routes: auth ─────────────────────────────────────────────────────────

@app.post("/api/signup")
def api_signup():
    payload = request.get_json(force=True)
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    name = (payload.get("name") or "").strip()
    country = (payload.get("country") or "").strip()
    language = payload.get("language") or "es"

    if not is_valid_email(email):
        return json_error("Email inválido")
    if len(password) < 6:
        return json_error("La contraseña debe tener al menos 6 caracteres")
    if not name:
        return json_error("El nombre es obligatorio")
    if language not in LANGUAGES:
        language = "es"

    now = datetime.now(timezone.utc).isoformat()
    with closing(get_db()) as conn:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            return json_error("Ya existe una cuenta con ese email")
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, name, country, language, avatar_seed, created_at) VALUES (?,?,?,?,?,?,?)",
            (email, hash_password(password), name, country, language, secrets.token_hex(4), now),
        )
        conn.commit()
        uid = cur.lastrowid

    session["user_id"] = uid
    return jsonify({"ok": True, "user": {"id": uid, "email": email, "name": name}})


@app.post("/api/login")
def api_login():
    payload = request.get_json(force=True)
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return json_error("Email o contraseña incorrectos", 401)
    session["user_id"] = row["id"]
    return jsonify({"ok": True, "user": {"id": row["id"], "email": row["email"], "name": row["name"]}})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/me")
@require_auth
def api_me(user):
    user = dict(user)
    user.pop("password_hash", None)
    return jsonify({"ok": True, "user": user})


# ── Routes: skills (browse / post) ──────────────────────────────────────

@app.get("/api/skills")
def api_skills_browse():
    q = (request.args.get("q") or "").strip()
    category = request.args.get("category") or ""
    with closing(get_db()) as conn:
        query = """
            SELECT skills.*, users.name as provider_name, users.country as provider_country,
                   users.language as provider_language, users.avatar_seed as provider_avatar
            FROM skills JOIN users ON skills.user_id = users.id
            WHERE skills.active = 1
        """
        params = []
        if category:
            query += " AND skills.category = ?"
            params.append(category)
        if q:
            query += " AND (skills.title LIKE ? OR skills.description LIKE ?)"
            like = f"%{q}%"
            params.extend([like, like])
        query += " ORDER BY skills.created_at DESC LIMIT 100"
        rows = conn.execute(query, params).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            rating_row = conn.execute(
                "SELECT AVG(rating) as avg_r, COUNT(*) as n FROM reviews WHERE reviewed_user_id=?",
                (d["user_id"],)
            ).fetchone()
            d["provider_rating"] = round(rating_row["avg_r"], 1) if rating_row["avg_r"] else None
            d["provider_review_count"] = rating_row["n"]
            results.append(d)
    return jsonify({"ok": True, "skills": results})


@app.post("/api/skills")
@require_auth
def api_skills_create(user):
    payload = request.get_json(force=True)
    title = (payload.get("title") or "").strip()
    category = payload.get("category") or ""
    description = (payload.get("description") or "").strip()
    price = payload.get("price", 0)
    currency = payload.get("currency", "USD")
    session_minutes = payload.get("session_minutes", 60)

    if not title or not description:
        return json_error("Título y descripción son obligatorios")
    if category not in CATEGORIES:
        return json_error("Categoría inválida")
    try:
        price = float(price)
        if price < 0:
            return json_error("El precio no puede ser negativo")
    except (TypeError, ValueError):
        return json_error("Precio inválido")
    try:
        session_minutes = int(session_minutes)
        if session_minutes <= 0:
            return json_error("La duración debe ser mayor a cero")
    except (TypeError, ValueError):
        return json_error("Duración inválida")

    now = datetime.now(timezone.utc).isoformat()
    with closing(get_db()) as conn:
        cur = conn.execute(
            "INSERT INTO skills (user_id, title, category, description, price, currency, session_minutes, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user["id"], title, category, description, price, currency, session_minutes, now),
        )
        conn.commit()
    return jsonify({"ok": True, "skill_id": cur.lastrowid})


@app.get("/api/skills/<int:skill_id>")
def api_skill_detail(skill_id):
    with closing(get_db()) as conn:
        row = conn.execute(
            """SELECT skills.*, users.name as provider_name, users.country as provider_country,
                      users.bio as provider_bio, users.language as provider_language
               FROM skills JOIN users ON skills.user_id = users.id WHERE skills.id=?""",
            (skill_id,)
        ).fetchone()
        if not row:
            return json_error("No encontrado", 404)
        d = dict(row)
        reviews = conn.execute(
            """SELECT reviews.*, users.name as reviewer_name FROM reviews
               JOIN requests ON reviews.request_id = requests.id
               JOIN users ON reviews.reviewer_id = users.id
               WHERE requests.skill_id = ? ORDER BY reviews.created_at DESC LIMIT 20""",
            (skill_id,)
        ).fetchall()
        d["reviews"] = [dict(r) for r in reviews]
    return jsonify({"ok": True, "skill": d})


@app.delete("/api/skills/<int:skill_id>")
@require_auth
def api_skill_delete(user, skill_id):
    with closing(get_db()) as conn:
        row = conn.execute("SELECT user_id FROM skills WHERE id=?", (skill_id,)).fetchone()
        if not row:
            return json_error("No encontrado", 404)
        if row["user_id"] != user["id"]:
            return json_error("No autorizado", 403)
        conn.execute("UPDATE skills SET active=0 WHERE id=?", (skill_id,))
        conn.commit()
    return jsonify({"ok": True})


# ── Routes: requests (booking a session) ────────────────────────────────

@app.post("/api/requests")
@require_auth
def api_request_create(user):
    payload = request.get_json(force=True)
    skill_id = payload.get("skill_id")
    message = (payload.get("message") or "").strip()
    if not skill_id or not message:
        return json_error("skill_id y message son obligatorios")

    with closing(get_db()) as conn:
        skill = conn.execute("SELECT * FROM skills WHERE id=? AND active=1", (skill_id,)).fetchone()
        if not skill:
            return json_error("Ese servicio no existe o ya no está disponible", 404)
        if skill["user_id"] == user["id"]:
            return json_error("No podés pedirte una sesión a vos mismo")

        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO requests (skill_id, requester_id, provider_id, initial_message, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (skill_id, user["id"], skill["user_id"], message, "pending", now, now),
        )
        req_id = cur.lastrowid
        conn.execute(
            "INSERT INTO messages (request_id, sender_id, text, created_at) VALUES (?,?,?,?)",
            (req_id, user["id"], message, now),
        )
        conn.commit()
    return jsonify({"ok": True, "request_id": req_id})


@app.get("/api/requests")
@require_auth
def api_requests_list(user):
    with closing(get_db()) as conn:
        sent = conn.execute(
            """SELECT requests.*, skills.title as skill_title, users.name as other_name, users.language as other_language
               FROM requests JOIN skills ON requests.skill_id = skills.id
               JOIN users ON requests.provider_id = users.id
               WHERE requests.requester_id=? ORDER BY requests.updated_at DESC""",
            (user["id"],)
        ).fetchall()
        received = conn.execute(
            """SELECT requests.*, skills.title as skill_title, users.name as other_name, users.language as other_language
               FROM requests JOIN skills ON requests.skill_id = skills.id
               JOIN users ON requests.requester_id = users.id
               WHERE requests.provider_id=? ORDER BY requests.updated_at DESC""",
            (user["id"],)
        ).fetchall()
    return jsonify({"ok": True, "sent": [dict(r) for r in sent], "received": [dict(r) for r in received]})


@app.post("/api/requests/<int:req_id>/respond")
@require_auth
def api_request_respond(user, req_id):
    payload = request.get_json(force=True)
    action = payload.get("action")
    if action not in ("accept", "decline", "complete"):
        return json_error("Acción inválida")
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()
        if not row:
            return json_error("No encontrado", 404)
        if action in ("accept", "decline") and row["provider_id"] != user["id"]:
            return json_error("No autorizado", 403)
        if action == "complete" and user["id"] not in (row["provider_id"], row["requester_id"]):
            return json_error("No autorizado", 403)
        new_status = {"accept": "accepted", "decline": "declined", "complete": "completed"}[action]
        conn.execute(
            "UPDATE requests SET status=?, updated_at=? WHERE id=?",
            (new_status, datetime.now(timezone.utc).isoformat(), req_id),
        )
        conn.commit()
    return jsonify({"ok": True, "status": new_status})


# ── Routes: messages (with AI translation) ──────────────────────────────

@app.get("/api/requests/<int:req_id>/messages")
@require_auth
def api_request_messages(user, req_id):
    with closing(get_db()) as conn:
        req_row = conn.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()
        if not req_row:
            return json_error("No encontrado", 404)
        if user["id"] not in (req_row["requester_id"], req_row["provider_id"]):
            return json_error("No autorizado", 403)
        msgs = conn.execute(
            "SELECT messages.*, users.name as sender_name FROM messages JOIN users ON messages.sender_id = users.id WHERE request_id=? ORDER BY messages.created_at ASC",
            (req_id,)
        ).fetchall()
    return jsonify({"ok": True, "messages": [dict(m) for m in msgs], "request": dict(req_row)})


@app.post("/api/requests/<int:req_id>/messages")
@require_auth
def api_request_message_send(user, req_id):
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    if not text:
        return json_error("Mensaje vacío")

    with closing(get_db()) as conn:
        req_row = conn.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()
        if not req_row:
            return json_error("No encontrado", 404)
        if user["id"] not in (req_row["requester_id"], req_row["provider_id"]):
            return json_error("No autorizado", 403)

        other_id = req_row["provider_id"] if user["id"] == req_row["requester_id"] else req_row["requester_id"]
        other = conn.execute("SELECT language FROM users WHERE id=?", (other_id,)).fetchone()
        other_lang = other["language"] if other else None

        translated = None
        if other_lang and other_lang != user["language"]:
            translated = translate_text(text, other_lang)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO messages (request_id, sender_id, text, translated_text, translated_for_lang, created_at) VALUES (?,?,?,?,?,?)",
            (req_id, user["id"], text, translated, other_lang, now),
        )
        conn.execute("UPDATE requests SET updated_at=? WHERE id=?", (now, req_id))
        conn.commit()
    return jsonify({"ok": True, "translated": translated})


# ── Routes: reviews ──────────────────────────────────────────────────────

@app.post("/api/reviews")
@require_auth
def api_review_create(user):
    payload = request.get_json(force=True)
    req_id = payload.get("request_id")
    rating = payload.get("rating")
    comment = (payload.get("comment") or "").strip()

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            return json_error("La calificación debe ser entre 1 y 5")
    except (TypeError, ValueError):
        return json_error("Calificación inválida")

    with closing(get_db()) as conn:
        req_row = conn.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()
        if not req_row:
            return json_error("No encontrado", 404)
        if user["id"] not in (req_row["requester_id"], req_row["provider_id"]):
            return json_error("No autorizado", 403)
        if req_row["status"] != "completed":
            return json_error("Solo se puede calificar una sesión completada")
        reviewed_id = req_row["provider_id"] if user["id"] == req_row["requester_id"] else req_row["requester_id"]
        existing = conn.execute("SELECT id FROM reviews WHERE request_id=?", (req_id,)).fetchone()
        if existing:
            return json_error("Ya calificaste esta sesión")
        conn.execute(
            "INSERT INTO reviews (request_id, reviewer_id, reviewed_user_id, rating, comment, created_at) VALUES (?,?,?,?,?,?)",
            (req_id, user["id"], reviewed_id, rating, comment, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    return jsonify({"ok": True})


# ── Boot ─────────────────────────────────────────────────────────────────

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
