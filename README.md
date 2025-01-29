# Outfit Analyzer

A real-time outfit analysis tool using computer vision and LangChain.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create .env file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the application:
```bash
python src/main.py
```

Controls:
- Press SPACE to capture and analyze an outfit
- Press Q to quit

## Output

Analysis results are saved in the `outfit_results` directory, including:
- Captured images (JPG)
- Analysis text files (TXT)