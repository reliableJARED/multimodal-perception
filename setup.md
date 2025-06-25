# Setup

## Requirements
- Python 3.8+
- GPU recommended for better performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/reliableJARED/multimodal-perception.git
cd multimodal-perception
```

2. Create and activate virtual environment:
```bash
python -m venv venv_mp

# Windows:
venv_mp\Scripts\activate

# Mac/Linux:
source venv_mp/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional models (if needed):
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/snakers4/silero-vad.git
```

## Note
Hugging Face models (like Whisper, Moondream2, etc.) will be automatically downloaded when first used. Make sure you have sufficient disk space and internet connection.

That's it! You're ready to start using the multimodal perception tools.