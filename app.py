from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import json
import datetime
import os

# ---------- Config ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/ai_learning")
KB_PATH = os.getenv("KB_PATH", "kb.json")
MATERIALS_PATH = os.getenv("MATERIALS_PATH", "materials.json")
CHAT_SIM_THRESHOLD = float(os.getenv("CHAT_SIM_THRESHOLD", 0.35))

app = Flask(__name__)
app.config["MONGO_URI"] = MONGO_URI
CORS(app, resources={r"/*": {"origins": "*"}})  # allow all origins for dev; scope down in production
mongo = PyMongo(app)

# ---------- Utilities to load JSON files ----------
def load_json_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Load knowledge base ----------
kb = load_json_file(KB_PATH)  # list of {"question":..., "answer":..., "tags":[...]}
kb_questions = [entry.get("question","") for entry in kb]
kb_answers = [entry.get("answer","") for entry in kb]

# TF-IDF vectorizer; if KB empty, keep things safe
vectorizer = TfidfVectorizer(stop_words="english")
if kb_questions:
    try:
        tfidf_kb = vectorizer.fit_transform(kb_questions)
    except Exception:
        # fallback: empty matrix
        tfidf_kb = None
else:
    tfidf_kb = None

# ---------- Load materials ----------
materials = load_json_file(MATERIALS_PATH)  # list of {"id":.., "title":.., "description":.., "tags":[..]}

# ---------- Endpoints ----------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.datetime.utcnow().isoformat()})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Request JSON:
      { "user_id": "user123", "message": "How to reset password?" }
    Response JSON:
      { "response": "...", "source":"kb"|"fallback", "top_matches": [ {question, score}, ... ] }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "anonymous")
    message = data.get("message", "")
    if not message or not isinstance(message, str):
        return jsonify({"error": "message is required and must be a string"}), 400

    response = "Sorry, I couldn't find an answer."
    source = "fallback"
    top_matches = []

    # If KB exists, compute similarity
    if tfidf_kb is not None:
        try:
            msg_vec = vectorizer.transform([message])
            sims = cosine_similarity(msg_vec, tfidf_kb).flatten()
            best_idx = int(sims.argmax()) if sims.size > 0 else -1
            best_score = float(sims[best_idx]) if best_idx >= 0 else 0.0

            # Prepare top matches
            top_n = min(3, len(sims))
            top_idx = sims.argsort()[::-1][:top_n]
            top_matches = [{"question": kb_questions[i], "score": float(sims[i])} for i in top_idx]

            if best_idx >= 0 and best_score >= CHAT_SIM_THRESHOLD:
                response = kb_answers[best_idx]
                source = "kb"
            else:
                # Friendly fallback with suggestions
                response = (
                    "I couldn't find a direct match in the knowledge base. "
                    "Try rephrasing, or here are related KB questions you can check."
                )
        except Exception as e:
            # If vectorization or similarity fails, return fallback but still log
            response = "Internal NLP error — fallback response."
            source = "fallback"

    # Save chat log
    try:
        mongo.db.chats.insert_one({
            "user_id": user_id,
            "message": message,
            "response": response,
            "source": source,
            "timestamp": datetime.datetime.utcnow()
        })
    except Exception:
        # ignore DB errors for now but don't crash the API
        pass

    return jsonify({"response": response, "source": source, "top_matches": top_matches})

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Request JSON:
      { "user_id":"user123", "num":5 }
    Response JSON:
      { "recommendations": [ {id,title,description,tags}, ... ] }
    Basic content-based scoring using tag overlap with user interactions.
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "anonymous")
    num = int(data.get("num", 5))

    # Gather user tags from stored interactions
    user_tags = set()
    try:
        interactions = mongo.db.user_interactions.find({"user_id": user_id})
        for it in interactions:
            tags = it.get("tags", [])
            for t in tags:
                user_tags.add(t)
    except Exception:
        interactions = []

    recs = []
    if user_tags:
        # score materials by tag overlap
        scored = []
        for m in materials:
            mtags = set(m.get("tags", []))
            overlap = len(user_tags.intersection(mtags))
            scored.append((overlap, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        recs = [m for score, m in scored if score > 0][:num]

    # If not enough recommendations, append most accessed/popular materials (fallback)
    if len(recs) < num:
        try:
            pipeline = [
                {"$group": {"_id": "$material_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": max(10, num)}
            ]
            pop = list(mongo.db.material_access.aggregate(pipeline))
            pop_ids = [p["_id"] for p in pop]
        except Exception:
            pop_ids = []

        # add popular items first
        for pid in pop_ids:
            for m in materials:
                if m["id"] == pid and m not in recs:
                    recs.append(m)
                    break
            if len(recs) >= num:
                break

        # fill from materials list if still short
        for m in materials:
            if m not in recs:
                recs.append(m)
            if len(recs) >= num:
                break

    return jsonify({"recommendations": recs[:num]})

@app.route("/interaction", methods=["POST"])
def interaction():
    """
    Record that a user viewed/read a material:
    Request JSON:
      { "user_id":"user123", "material_id":"m001" }
    This stores a material access document and a lightweight user_interaction (with tags).
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "anonymous")
    material_id = data.get("material_id")
    if not material_id:
        return jsonify({"error": "material_id required"}), 400

    mat = next((m for m in materials if m["id"] == material_id), None)
    tags = mat.get("tags", []) if mat else []

    now = datetime.datetime.utcnow()
    try:
        mongo.db.material_access.insert_one({
            "user_id": user_id,
            "material_id": material_id,
            "timestamp": now
        })
        mongo.db.user_interactions.insert_one({
            "user_id": user_id,
            "material_id": material_id,
            "tags": tags,
            "timestamp": now
        })
    except Exception:
        # ignore DB insert errors so user doesn't get 500 for transient DB issues
        pass

    return jsonify({"ok": True})

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # debug=True for development only
    app.run(host="0.0.0.0", port=port, debug=True)
