"""Explicit synthetic fixtures. This module is never used by real mode."""
import base64
import io
import time

from PIL import Image, ImageDraw


def frame(timestamp: float, number: int) -> dict:
    image = Image.new("RGB", (720, 420), "#eeecfa")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 36, 684, 384), radius=24, outline="#7565df", width=3)
    draw.text((64, 80), "DEMO / SYNTHETIC FRAME", fill="#27213e", font_size=28)
    draw.text((64, 145), f"Event {number} - {timestamp:.1f} seconds", fill="#51466e", font_size=24)
    draw.text((64, 225), "Interface test only. Not evidence from this video.", fill="#51466e", font_size=18)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82)
    return {"timestamp_sec": timestamp, "image_base64": base64.b64encode(output.getvalue()).decode("ascii")}


def analyze(scenario: str, delay: float) -> dict:
    time.sleep(delay)
    if scenario == "error":
        raise RuntimeError("Explicit mock failure")
    if scenario == "clean":
        return {"verdict": "no_violation", "violations": []}
    events = [{"type": "Учебный пример: момент для проверки", "timecode_sec": 1.5,
               "description": "Тестовый сценарий интерфейса. Это вымышленное событие; кадры созданы программно и не относятся к загруженному видео. Правовой вывод и нормативный источник отсутствуют.",
               "evidence_frames": [frame(1, 1), frame(1.5, 1), frame(2, 1)]}]
    if scenario == "multiple":
        events.append({"type": "Учебный пример: второе событие", "timecode_sec": 3,
                       "description": "Второе вымышленное событие для проверки нескольких карточек. Настоящий анализатор передаст полное описание наблюдений и проверенную нормативную ссылку.",
                       "evidence_frames": [frame(3, 2)]})
    return {"verdict": "violation", "violations": events}
