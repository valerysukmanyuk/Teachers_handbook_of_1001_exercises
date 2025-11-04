
---

# 🧠 Teacher's Handbook of 1001 Exercises

> An intelligent tool for automatically generating **fill-in-the-gap** vocabulary exercises from video content.

---

## Overview

**Teacher’s Handbook of 1001 Exercises** is an AI-powered assistant that helps educators create contextualized language exercises directly from video materials.
Using **Whisper** for transcription, **spaCy** for linguistic analysis, and **LangChain** for intelligent reasoning, the system automatically selects meaningful vocabulary and generates well-formatted *fill-in-the-gap* tasks suitable for different learner levels.

This tool turns authentic video input into ready-to-use language exercises — saving teachers hours of manual work while keeping tasks pedagogically sound.

Built with optimization for **Intel® NPU** (Neural Processing Unit), the tool can leverage **intel-npu-acceleration** for faster local inference.
If no NPU is available, it will still run normally using CPU.

---

## Key Features

* **Automatic transcription** of video/audio using [OpenAI Whisper](https://github.com/openai/whisper)
* **Fill-in-the-gap exercise generation** from the transcribed text
* **Linguistic analysis** with [spaCy](https://spacy.io) to identify meaningful vocabulary using POS tagging
* **Autonomous task creation agent** built with [LangChain](https://www.langchain.com)
* **Adaptive vocabulary selection** — selects words based on learner level and contextual importance
* **Consistent exercise formatting** ready for export or integration with learning platforms
* **Intel® NPU optimization** — accelerated performance on compatible hardware via *intel-npu-acceleration*

---

## Tech Stack

| Component                  | Purpose                                                  |
| -------------------------- | -------------------------------------------------------- |
| **LangChain + DeepSeek**   | Agent framework for decision making and context analysis |
| **Whisper**                | Speech-to-text transcription                             |
| **spaCy**                  | Tokenization, POS tagging, and vocabulary filtering      |
| **intel-npu-acceleration** | Optional Intel® NPU runtime for hardware acceleration    |

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/valerysukmanyuk/Teacher-s_handbook_of_1001_exercises.git
   cd Teacher-s_handbook_of_1001_exercises
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install Intel NPU acceleration library:**

   ```bash
   pip install intel-npu-acceleration
   ```

   This enables optimized execution on devices equipped with Intel® NPU.
   The tool will still work on CPU/GPU even if this library is not installed.

4. **Install FFmpeg (required for Whisper):**

   Whisper relies on [FFmpeg](https://ffmpeg.org/) to process audio and video files.

   * **Windows (via Chocolatey):**

     ```bash
     choco install ffmpeg
     ```

   * **macOS (via Homebrew):**

     ```bash
     brew install ffmpeg
     ```

   * **Linux (Debian/Ubuntu):**

     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```

5. **Prepare your media files:**

   Place your **video** or **audio** files in the same folder as `main.py`.
   Example:

   ```
   Teacher-s_handbook_of_1001_exercises/
   ├── main.py
   ├── agent.py
   ├── stopwords.txt
   ├── requirements.txt
   └── video_lesson.mp4   ← your input file
   ```

6. **Run the main script:**

   ```bash
   python main.py
   ```

   The tool will automatically detect and process the video/audio file to generate a fill-in-the-gap exercise.
   If available, Intel® NPU acceleration will be used automatically.

---

## Usage

The tool automatically processes a video file or video link and produces a **fill-in-the-gap exercise** based on transcribed and analyzed text.

### Basic workflow

1. The user provides a video file (or audio source).
2. Whisper transcribes the content.
3. spaCy processes the text to extract meaningful words based on POS tagging.
4. The LangChain agent evaluates context and level appropriateness.
5. The output is a formatted cloze exercise.

---

## Example Output Format

```text
educational_video_01.mp4

1. Learning [English] helps people connect across cultures.
2. ...
```

---

## License

This project is distributed under the **MIT License**.
Feel free to use, modify, and contribute to improve it.
