# Chess AI REST API Service: Integration & Deployment Guide

This guide details how to activate the Python environment, run the high-concurrency Chess AI API on port `3210`, and integrate it into your backend or local application.

---

## 1. Sourcing the Virtual Environment

Before running the API service, you must activate the Python environment containing all necessary packages (PyTorch, FastAPI, Uvicorn, Python-Chess, NumPy).

### Option A: Sourcing your PyCharm Virtual Environment (Recommended)
Since your primary virtual environment resides in your active PyCharm folder, activate it by running:
```bash
source /home/thanglm2006/PycharmProjects/alphaOne/.venv/bin/activate
```

### Option B: Sourcing inside the ChessAlpha Directory
If you have a `venv` or `env` directory in your current workspace `/home/thanglm2006/ChessAlpha`, activate it using:
```bash
source venv/bin/activate
```

---

## 2. Launching the API Service

I have provided a self-cleaning, robust startup script `run_api.sh` in the workspace root. 

### Step 1: Ensure the script is executable
```bash
chmod +x run_api.sh
```

### Step 2: Run the script
To launch the service on the default port **`3210`**:
```bash
./run_api.sh
```

*Note: The script automatically checks if port `3210` is occupied by zombie processes, terminates them safely, activates the virtual environment, and boots the FastAPI server using Uvicorn.*

### Advanced Configurations
You can adjust the port and the maximum concurrent GPU/CPU operations (concurrency limit) using environment variables:
```bash
# Run on port 8080 allowing up to 4 concurrent heavy MCTS operations on the GPU
CHESS_CONCURRENT_TASKS=4 ./run_api.sh 8080
```

---

## 3. API Endpoints Reference

Once running, you can explore the interactive API dashboard (OpenAPI / Swagger UI) directly in your browser at:
👉 **[http://localhost:3210/docs](http://localhost:3210/docs)**

### A. Check Status & Concurrency
* **Method:** `GET`
* **Route:** `/api/status`
* **Purpose:** Checks server status, GPU device configuration, loaded model count, and concurrent queue limits.
* **Sample Response:**
  ```json
  {
    "status": "online",
    "device": "cuda",
    "total_models": 3,
    "default_model": "best_model",
    "cuda_available": true,
    "max_concurrent_tasks": 2
  }
  ```

### B. List Available Chess Checkpoints
* **Method:** `GET`
* **Route:** `/api/models`
* **Purpose:** Fetch loaded AI checkpoints from the `checkpoints/` directory to allow players to select models.
* **Sample Response:**
  ```json
  {
    "models": [
      {
        "key": "best_model",
        "display": "V1 Best  (Plain ResNet)",
        "arch": "Plain ResNet 10×192",
        "arch_type": "v1",
        "category": "AlphaZero V1 (Pretrained 20M)",
        "filename": "best_model.pth",
        "epoch": 25,
        "loss": 0.0412
      },
      {
        "key": "model_ep5",
        "display": "SE-ResNet EP5",
        "arch": "SE-ResNet 12×192",
        "arch_type": "se",
        "category": "SE-ResNet (Failed Transfer)",
        "filename": "model_ep5.pth",
        "epoch": 5,
        "loss": 0.125
      }
    ],
    "default": "best_model"
  }
  ```

### C. Request AI Move (Stateless playing)
* **Method:** `POST`
* **Route:** `/api/move`
* **Purpose:** Submit player's move or full board status, and receive the calculated AI move response.
* **Request Body:**
  ```json
  {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "player_move": "e4",            // Supports UCI ("e2e4") or SAN ("e4")
    "model": "best_model",          // Checkpoint key
    "difficulty": 3                 // 1 (Random), 2 (Medium), 3 (Hard), 4 (MCTS Expert)
  }
  ```
* **Response Body:**
  ```json
  {
    "player_move": "e2e4",
    "player_move_san": "e4",
    "ai_move": "e7e5",
    "ai_move_san": "e5",
    "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "is_game_over": false,
    "result": null,
    "termination": null,
    "evaluation": 0.045,
    "top_moves": [
      { "uci": "e7e5", "san": "e5", "prob": 0.42 },
      { "uci": "c7c5", "san": "c5", "prob": 0.31 }
    ]
  }
  ```

### D. Request AI Move First
* **Method:** `POST`
* **Route:** `/api/ai-move-first`
* **Purpose:** Triggered when the AI plays as White and makes the opening move or is configured to move first on a custom board.
* **Request Body:**
  ```json
  {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "model": "best_model",
    "difficulty": 3
  }
  ```

---

## 4. Backend Code Integration Examples

Here are concrete code snippets demonstrating how to interact with the API service from different tech stacks.

### Python Backend Integration (Flask, Django, FastAPI, or Script)
```python
import requests

API_URL = "http://localhost:3210/api"

# 1. Fetch available models for selection
def get_models():
    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        data = response.json()
        print("Available Models:")
        for model in data["models"]:
            print(f"- {model['key']}: {model['display']}")
        return data["default"]
    return None

# 2. Request a move (stateless turn passing)
def make_ai_move(fen, player_move, model="best_model", difficulty=3):
    payload = {
        "fen": fen,
        "player_move": player_move,
        "model": model,
        "difficulty": difficulty
    }
    
    # Post request will yield the updated state and the AI response
    response = requests.post(f"{API_URL}/move", json=payload)
    if response.status_code == 200:
        res_data = response.json()
        print(f"Player Move (SAN): {res_data['player_move_san']}")
        print(f"AI Response (SAN): {res_data['ai_move_san']} (UCI: {res_data['ai_move']})")
        print(f"New FEN State    : {res_data['fen']}")
        print(f"Is Game Over     : {res_data['is_game_over']}")
        return res_data
    else:
        print(f"Error: {response.text}")
        return None

# Example Usage
if __name__ == "__main__":
    # Get current default model
    default_model = get_models()
    
    # Send player move "e4" on the starting board
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    make_ai_move(starting_fen, player_move="e4", model=default_model, difficulty=3)
```

### JavaScript / Node.js Backend Integration
```javascript
const axios = require('axios'); // or use native fetch

const API_BASE = 'http://localhost:3210/api';

// Request AI move on player turn
async function sendPlayerMove(currentFen, playerMove, modelKey = 'best_model', difficulty = 3) {
    try {
        const response = await axios.post(`${API_BASE}/move`, {
            fen: currentFen,
            player_move: playerMove,
            model: modelKey,
            difficulty: difficulty
        });

        const data = response.data;
        if (data.is_game_over) {
            console.log(`Game ended by ${data.termination}. Result: ${data.result}`);
        } else {
            console.log(`AI Move: ${data.ai_move_san} (${data.ai_move})`);
            console.log(`Updated FEN: ${data.fen}`);
        }
        return data;
    } catch (error) {
        console.error('API Error:', error.response ? error.response.data : error.message);
    }
}

// Request AI to move first as White
async function requestAiFirst(startingFen, modelKey = 'best_model', difficulty = 3) {
    try {
        const response = await axios.post(`${API_BASE}/ai-move-first`, {
            fen: startingFen,
            model: modelKey,
            difficulty: difficulty
        });
        
        console.log(`AI First Move: ${response.data.ai_move_san}`);
        return response.data;
    } catch (error) {
        console.error('API Error:', error.message);
    }
}
```
