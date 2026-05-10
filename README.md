# Cathey — Offline Smart Home Assistant

EECS 6895 Final Project · Columbia University

Cathey is a fully offline voice assistant deployable on a **Raspberry Pi 5**.
It listens for the wake word "Cathey", classifies user intent into 4 categories, executes smart home commands on real GPIO hardware, and learns user preferences over time using a four-layer memory system.

**No cloud. No internet. Everything runs on the Pi.**

---

## Project Structure

```
6895-Final-Project-Cathey/
├── cathey.py               # Entry point — runs the full assistant in a VAD loop
├── config.py               # All constants (model names, audio params, GPIO pins, thresholds)
├── schema.py               # Device schema, command validation, execution
├── llm_parser.py           # LLM loading + 3 inference methods (parse / followup / qa)
├── memory.py               # Four-layer memory (working/episodic/semantic/procedural)
├── agent.py                # CatheyAgent — intent routing + dialogue state machine
├── audio.py                # STT (faster-whisper) + TTS (Piper) + VAD audio listener
├── rule_based.py           # Regex fast path for direct commands (state-aware)
├── gpio_executor.py        # Hardware control (LED ring, fan, dual stepper motors)
├── benchmark_quantization.py  # 5-variant GGUF benchmark on Pi 5
├── deploy.sh               # rsync + model download
├── cathey.service          # systemd unit
├── finetune/
│   ├── train_data.py            # 22500 (utterance, JSON) pairs
│   ├── generate_train_data.py   # Dataset generator (templates + script expansion)
│   └── lora_training.ipynb      # LoRA fine-tuning notebook
└── tests/
    ├── software_test.ipynb      # Text-level tests + batch evaluation
    └── dev_debug.ipynb          # Microphone tests + continuous VAD loop
```

---

## Intent Categories

| Category | Trigger | Example |
|---|---|---|
| `direct_command` | Device + explicit action | "Cathey, turn on the light." |
| `needs_clarification` | Vague feeling / atmosphere request | "Cathey, it's a bit dark." |
| `general_qa` | Non-device question | "Cathey, how do I store leftovers?" |
| `invalid` | No "Cathey" wake word | "Turn on the light." |

---

## Four-Layer Memory

| Layer | Storage | Lifetime | Purpose |
|---|---|---|---|
| Working | RAM `deque` | Current session | Last 8 turns of conversation |
| Episodic | ChromaDB (local) | Persistent | RAG retrieval of similar past interactions |
| Semantic | JSON file | Persistent | User preferences (e.g. preferred AC temp) |
| Procedural | JSON file | Persistent | Learned trigger → action patterns (skip re-asking) |

Cosine similarity in procedural memory uses `all-MiniLM-L6-v2` (384-d) with threshold 0.92.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Yiwen543/6895-Final-Project-Cathey.git
cd 6895-Final-Project-Cathey
```

### 2. Create a Python environment

```bash
python -m venv cathey_env
source cathey_env/bin/activate            # Windows: cathey_env\Scripts\activate
```

### 3. Install dependencies

On a development machine (Mac / Linux / Windows):

```bash
pip install faster-whisper transformers peft trl datasets \
            sentence-transformers chromadb sounddevice soundfile \
            numpy
```

On the Raspberry Pi 5, also install:

```bash
pip install -r requirements_pi.txt
# llama-cpp-python (with OpenBLAS), pi5neo, lgpio, piper-tts
```

### 4. Download the LLM

The default model is **Qwen2.5-3B-Instruct in GGUF Q3_K_M format** (~1.5 GB). Download once:

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
    qwen2.5-3b-instruct-q3_k_m.gguf \
    --local-dir models --local-dir-use-symlinks False
```

Path is set in `config.py`:

```python
LLM_MODEL_PATH = "models/qwen2.5-3b-instruct-q3_k_m.gguf"
```

### 5. Run the assistant

```bash
python cathey.py
```

Cathey loads all models, then enters a continuous VAD loop. Speak `"Cathey, ..."` followed by your request. Press `Ctrl+C` to stop.

A short demo session:

```
User:   Cathey, turn on the light.
Cathey: Sure, turning on the light!
        (LED ring → full white)

User:   Cathey, make the light warmer.
Cathey: Setting warm light.
        (LED ring → warm white, level 4)

User:   Cathey, open the curtain a little.
Cathey: Opening the curtain a little.
        (Stepper motor → 20% open)

User:   Cathey, I feel cold.
Cathey: Would you like me to close the window or
        raise the AC temperature?
User:   Close the window.
Cathey: Closing the window now.
        (Window stepper motor → 0%)
```

---

## Running Tests

### Text tests (no microphone needed)

```bash
jupyter notebook tests/software_test.ipynb
```

**What runs:**
- **Group A** — Regression cases for all 4 intent types + hard cases (colloquial inputs, food-safety questions).
- **Group B** — Stateful demo of procedural memory: first time → clarification asked; second time → auto-resolved.
- **Batch evaluation** — Accuracy table + average latency.

Each test resets dialogue state before running so cases are independent.

### Audio tests (microphone required)

```bash
jupyter notebook tests/dev_debug.ipynb
```

**What runs:**
- Single fixed-duration recording → STT → agent → result.
- Continuous VAD loop: speak naturally, Cathey captures full utterances automatically. `Ctrl+C` to stop.

---

## Quantization Benchmark

We compared 5 GGUF variants of Qwen2.5-Instruct on the same Pi 5 with identical settings:

| Model | Type Acc | Cmd Acc | Avg latency | Size |
|---|---|---|---|---|
| 1.5B-Q3_K_M | 15% | 20% | 5007 ms | 0.8 GB |
| 1.5B-Q4_K_M | 30% | 0% | 2728 ms | 1.1 GB |
| 1.5B-Q5_K_M | 50% | 80% | 4028 ms | 1.3 GB |
| **3B-Q3_K_M** | **85%** | 80% | **3887 ms** | **1.5 GB** |
| 3B-Q4_K_M | 75% | 100% | 5712 ms | 2.0 GB |

**3B-Q3_K_M wins** on the type-accuracy / latency / size trade-off. To reproduce:

```bash
python benchmark_quantization.py
```

Test cases live in `benchmark_quantization.py` (5 per intent class × 4 classes = 20 total). Results are written to `benchmark_results.csv` and `benchmark_results.md`.

---

## LoRA Fine-Tuning

Fine-tuning teaches the model to reliably output valid JSON and correctly classify hard cases that the base model gets wrong (e.g. colloquial atmosphere requests, food-safety questions).

```bash
jupyter notebook finetune/lora_training.ipynb
```

**Configuration:** rank `r=8`, `alpha=16`, 7 target modules (~0.44% trainable params), 3 epochs, batch size 1, gradient accumulation 8, LR `2e-4`, cosine warmup, max sequence length 512. TRL `SFTTrainer` applies response-only loss masking.

**Training time:**

| Hardware | Time per epoch |
|---|---|
| CPU (Mac / Pi 5) | ~15–30 min |
| GPU (CUDA) | ~1–3 min |

**To regenerate the training set** (or scale it differently):

```bash
python finetune/generate_train_data.py
```

This writes `finetune/train_data.py` with 22,500 examples by combining hand-written seed templates and a small expansion script that varies wake-word forms, parameter values, and paraphrasings.

---

## Hardware

| Device | Pin / Bus | Library | Notes |
|---|---|---|---|
| WS2812B 12-LED ring | SPI0 (MOSI = GPIO 10) | `pi5neo` | 5 color temps + RGB cycle |
| 4-pin PWM fan (AC sim) | GPIO 12 (PWM0, 1 kHz) | `lgpio` | 16°C → 100%, 30°C → 10% |
| 28BYJ-48 stepper #1 (curtain) | GPIO 5 / 6 / 13 / 26 | `lgpio` | Tracks position in steps |
| 28BYJ-48 stepper #2 (window) | GPIO 17 / 27 / 22 / 23 | `lgpio` | Open/close two-state |
| USB microphone | USB | `sounddevice` | 16 kHz sample rate |
| Bluetooth speaker | Bluetooth | `pw-play` | Piper TTS output |

**Color temperature scale:**

| Level | K | Label | RGB |
|---|---|---|---|
| 1 | 6500 | daylight | (255, 255, 255) |
| 2 | 5000 | reading | (255, 255, 230) |
| 3 | 4000 | neutral | (255, 255, 200) |
| 4 | 3000 | warm | (255, 197, 143) |
| 5 | 2700 | candlelight | (255, 147, 41) |

---

## Deployment on Raspberry Pi 5

### Transfer the project

```bash
rsync -av --exclude 'cathey_env' --exclude '__pycache__' \
    ./ pi@raspberrypi.local:~/cathey/
```

Or use the included `deploy.sh`.

### Run as a systemd service

`cathey.service` is included. Install it once on the Pi:

```bash
sudo cp cathey.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cathey
```

Live logs:

```bash
sudo journalctl -u cathey -f
```

---

## Configuration Reference

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL_PATH` | `models/qwen2.5-3b-instruct-q3_k_m.gguf` | LLM (GGUF) |
| `LLM_N_CTX` | `2048` | Context window |
| `LLM_MAX_NEW_TOKENS` | `150` | Max tokens per generation |
| `WHISPER_MODEL_SIZE` | `tiny.en` | STT model size |
| `ENERGY_THRESHOLD` | `0.05` | VAD silence cutoff |
| `WAKE_WORD_FUZZY_THRESHOLD` | `0.78` | Fuzzy wake-word similarity cutoff |
| `SKILL_SIM_THRESHOLD` | `0.92` | Procedural memory cosine similarity cutoff |
| `WORKING_MAXLEN` | `8` | Max turns in working memory |
| `LORA_R` | `8` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA scaling factor |

---

## How Each Module Connects

```
cathey.py
    │
    ├── LLMParser (llm_parser.py)
    │       ├── parse_unified()       ← classify intent → JSON
    │       ├── resolve_followup()    ← resolve clarification reply → JSON
    │       └── answer_qa()           ← RAG-augmented plain-text answer
    │
    ├── try_rule_based (rule_based.py)
    │       ← regex + state-aware relative adjustment
    │
    ├── MemoryManager (memory.py)
    │       ├── push_working()       ← append to session RAM
    │       ├── save_episode()       ← write to ChromaDB
    │       ├── update_pref()        ← write to user_prefs.json
    │       ├── record_skill()       ← write to skills.json
    │       ├── lookup_skill()       ← cosine search in procedural memory
    │       └── build_context()     ← aggregate all layers → RAG prompt
    │
    ├── CatheyAgent (agent.py)
    │       └── handle(text)         ← routes text through the full pipeline
    │
    ├── GPIOExecutor (gpio_executor.py)
    │       ├── execute(action)      ← dispatch to LED / fan / steppers
    │       └── get_device_state()   ← real-time read-back for state-aware rules
    │
    └── AudioListener (audio.py)
            ├── run_one_round()      ← fixed-duration recording
            └── continuous_loop()    ← VAD-gated streaming loop
```

---

## Troubleshooting

**Model file not found**
→ Check `LLM_MODEL_PATH` in `config.py` and that the GGUF file is downloaded to `models/`.

**`No module named 'sounddevice'`**
→ On Linux/Pi: `sudo apt install portaudio19-dev` then `pip install sounddevice`.

**Piper TTS has no audio output on Pi**
→ Check `aplay -l` to verify the audio device. For Bluetooth: `bluetoothctl` to pair, then test with `pw-play`.

**VAD captures too much background noise**
→ Increase `ENERGY_THRESHOLD` in `config.py` (try `0.05` – `0.10`).

**LLM latency too high**
→ Confirm `n_threads=4` matches Pi 5 cores. Try a smaller GGUF if needed (1.5B-Q5_K_M trades accuracy for speed).

**ChromaDB error on first run**
→ The `cathey_memory/` directory is created automatically. If corrupted, delete it: `rm -rf cathey_memory/`.

**Wake word detection misses common phrases**
→ Lower `WAKE_WORD_FUZZY_THRESHOLD` (e.g. from 0.78 to 0.70). Verify the variant list in `config.py` includes how you say "Cathey".
