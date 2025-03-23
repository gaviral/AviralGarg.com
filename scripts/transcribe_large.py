# In your terminal:
# pip uninstall numpy
# pip install numpy==1.24.3
# pip install --upgrade torch openai-whisper

import whisper
import os
model = whisper.load_model("large")

print(f"PWD: {os.getcwd()}")
result = model.transcribe("notes/reinforcement/rl1_part_2.mp4")

# save the transcription to notes/reinforcement/rl1_part1.txt
with open("notes/reinforcement/rl1_part_2_large.txt", "w") as f:
    f.write(result["text"])
