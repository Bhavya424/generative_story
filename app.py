from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Hugging Face API
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

client = InferenceClient(api_key=HF_API_TOKEN)

def call_hf(prompt, model=HF_MODEL_ID):
    if not HF_API_TOKEN or "your_token" in HF_API_TOKEN:
        return "Error: Hugging Face API Token is missing or invalid in .env"
    try:
        # Use a system message if the model supports it (most modern ones do)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional film production AI. You output ONLY valid JSON. No talk, no markdown, no code blocks, just pure JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        err_msg = str(e)
        app.logger.error(f"HF API Error: {err_msg}")
        if "403" in err_msg:
            return "Error: 403 Forbidden. Your Hugging Face token might lack 'Inference' permissions. Please check your token settings at huggingface.co/settings/tokens"
        return f"Error calling Hugging Face API: {err_msg}"

def extract_json(text):
    """Ultra-robust JSON extraction from potentially messy LLM output."""
    if not text: return None
    text = text.strip()
    
    # 1. Try direct parse first
    try:
        data = json.loads(text)
        return _deep_clean(data)
    except:
        pass

    # 2. Try to find bracketed segments manually to avoid regex greediness
    def find_json_segments(s):
        segments = []
        stack = []
        start_char = None
        start_idx = -1
        
        for i, char in enumerate(s):
            if char in '{[':
                if not stack:
                    start_char = char
                    start_idx = i
                stack.append(char)
            elif char in '}]':
                if stack:
                    top = stack.pop()
                    if (top == '{' and char == '}') or (top == '[' and char == ']'):
                        if not stack:
                            segments.append(s[start_idx:i+1])
                    else:
                        # Mismatched bracket, reset
                        stack = []
        return segments

    potentials = find_json_segments(text)
    # Sort by length descending, try longest first
    potentials.sort(key=len, reverse=True)
    
    for pot in potentials:
        try:
            data = json.loads(pot)
            return _deep_clean(data)
        except:
            # Try to fix common issues: unescaped quotes in strings
            # This is risky but we can try simple comma removal
            import re
            cleaned = re.sub(r',\s*([\]}])', r'\1', pot)
            try:
                data = json.loads(cleaned)
                return _deep_clean(data)
            except:
                continue
                
    return None

def _deep_clean(data):
    """Recursively find and parse stringified JSON within a data structure."""
    if isinstance(data, list):
        return [_deep_clean(item) for item in data]
    elif isinstance(data, dict):
        return {k: _deep_clean(v) for k, v in data.items()}
    elif isinstance(data, str):
        s = data.strip()
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
            try:
                nested = json.loads(s)
                return _deep_clean(nested)
            except:
                return data
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/list-models")
def list_models():
    # Providing a list of popular Hugging Face models for text generation
    models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "google/gemma-7b-it",
        "microsoft/Phi-3-mini-4k-instruct"
    ]
    return jsonify({"available_models": models})


@app.route("/api/generate-screenplay", methods=["POST"])
def generate_screenplay():
    data = request.json
    idea = data.get("idea", "")
    genre = data.get("genre", "Drama")
    tone = data.get("tone", "Serious")
    pages = data.get("pages", 5)

    prompt = f"""You are a professional Hollywood screenplay writer. 
    Write a compelling {pages}-page screenplay excerpt based on this idea: "{idea}"
    Genre: {genre}
    Tone: {tone}
    
    Format it properly with:
    - FADE IN:
    - Scene headings (INT./EXT.)
    - Action lines
    - Character names centered
    - Dialogue
    - Scene transitions
    
    Make it cinematic, emotionally resonant, and professional quality."""

    screenplay = call_hf(prompt)
    return jsonify({"screenplay": screenplay, "status": "success"})


@app.route("/api/generate-characters", methods=["POST"])
def generate_characters():
    data = request.json
    idea = data.get("idea", "")
    num_characters = data.get("num_characters", 3)

    prompt = f"""You are a master character designer for films.
    Create {num_characters} detailed character profiles for this story concept: "{idea}"
    
    Return a JSON array of objects. Each object MUST have these exact keys:
    "name", "role", "age", "background", "personality", "motivation", "arc", "quirks".
    
    IMPORTANT: "personality" should be a simple array of strings like ["trait1", "trait2"].
    
    Return ONLY valid JSON. No conversational filler, no markdown blocks."""

    result = call_hf(prompt)
    characters = extract_json(result)
    
    if characters is None:
        characters = [{"name": "Error or Parse Issue", "role": "N/A", "age": "N/A",
                       "background": result, "personality": [], 
                       "motivation": "", "arc": "", "quirks": ""}]
    
    return jsonify({"characters": characters, "status": "success"})


@app.route("/api/generate-production-plan", methods=["POST"])
def generate_production_plan():
    data = request.json
    idea = data.get("idea", "")
    budget = data.get("budget", "Medium")
    duration = data.get("duration", "Feature Film")

    prompt = f"""You are an experienced film producer and production manager.
    Create a detailed pre-production plan for this film concept: "{idea}"
    Budget Level: {budget}
    Film Type: {duration}
    
    Return ONLY a valid JSON object with these exact keys:
    - "timeline": array of {{"phase", "duration", "tasks"}} objects
    - "locations": array of {{"name", "type", "requirements"}} objects  
    - "crew": array of {{"role", "count", "notes"}} objects
    - "equipment": array of {{"category", "items"}} objects
    - "budget_breakdown": array of {{"category", "percentage", "notes"}} objects
    - "sound_design": object with {{"music_style", "sound_effects", "recording_notes", "post_production"}}
    
    Return ONLY valid JSON. No conversational filler, no markdown blocks."""

    result = call_hf(prompt)
    print(f"DEBUG Production Plan RAW: {result[:500]}...")
    plan = extract_json(result)
    
    if plan is None:
        plan = {"error": "Could not parse plan", "raw": result}
    
    return jsonify({"plan": plan, "status": "success"})


@app.route("/api/generate-sound-design", methods=["POST"])
def generate_sound_design():
    data = request.json
    idea = data.get("idea", "")
    genre = data.get("genre", "Drama")
    mood = data.get("mood", "Dramatic")

    prompt = f"""You are an award-winning sound designer and music supervisor.
    Create a comprehensive sound design document for this film: "{idea}"
    Genre: {genre}
    Mood/Tone: {mood}
    
    Return ONLY a valid JSON object with these exact keys:
    - "score": {{"style", "influences", "instruments", "themes"}} (themes is array)
    - "sound_palette": array of {{"scene_type", "sounds", "atmosphere"}}
    - "music_cues": array of {{"moment", "track_description", "emotion", "tempo"}}
    - "foley": array of {{"category", "key_sounds"}}
    - "technical_specs": {{"format", "sample_rate", "channels", "delivery"}}
    
    Return ONLY valid JSON. No conversational filler, no markdown blocks."""

    result = call_hf(prompt)
    print(f"DEBUG Sound Design RAW: {result[:500]}...")
    sound = extract_json(result)
    
    if sound is None:
        sound = {"error": "Could not parse", "raw": result}
    
    return jsonify({"sound_design": sound, "status": "success"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
