#!/usr/bin/env python3
"""
Модуль захвата звука с микрофона для SUDARSHANchakra
Использует sounddevice для записи в реальном времени с единой
предобработкой, совместимой с обученной моделью.
"""

import numpy as np
import sounddevice as sd
import torch
import time
from collections import deque
from pathlib import Path

# Добавляем корень проекта в путь для импортов
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from src.data_loader import AudioTransform


class AudioCapture:
    """
    Захватывает звук с микрофона через sounddevice и преобразует
    в Mel-спектрограммы, совместимые с моделью (нормализация min-max).
    """

    def __init__(self,
                 sample_rate: int = Config.SAMPLE_RATE,
                 duration: float = Config.DURATION,
                 device_index: int = None):
        """
        Args:
            sample_rate: Частота дискретизации (Гц)
            duration: Длина анализируемого отрезка (сек)
            device_index: Индекс устройства (None = устройство по умолчанию)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)

        # Единый преобразователь аудио -> спектрограмма (использует те же параметры, что и при обучении)
        self.transform = AudioTransform(
            sample_rate=sample_rate,
            duration=duration,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX
        )

        # Буфер для накопления аудио
        self.audio_buffer = deque(maxlen=self.target_samples)

        # Выводим список устройств
        print("[AUDIO] Доступные устройства:")
        print(sd.query_devices())

        # Устанавливаем устройство
        if device_index is not None:
            sd.default.device = device_index
            
        default_in = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device
        device_info = sd.query_devices(default_in, kind='input')
        print(f"[AUDIO] Используется микрофон: {device_info['name']} (индекс {default_in})")

        self.stream = None

    def start_stream(self):
        """Запускает поток захвата звука."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024
        )
        self.stream.start()
        print(f"[AUDIO] Поток запущен (частота: {self.sample_rate} Гц, длительность: {self.duration} сек)")

    def _audio_callback(self, in_data, frames, time_info, status):
        """Колбэк при поступлении новых данных."""
        if status:
            print(f"[AUDIO] Статус: {status}")

        audio_chunk = in_data.flatten()
        self.audio_buffer.extend(audio_chunk)

    def get_audio_segment(self):
        """
        Возвращает текущий сегмент аудио длительностью duration
        в формате (numpy array, sample_rate).
        """
        if len(self.audio_buffer) < self.target_samples:
            audio = np.zeros(self.target_samples, dtype=np.float32)
        else:
            audio = np.array(list(self.audio_buffer)[-self.target_samples:], dtype=np.float32)

        return audio, self.sample_rate

    def get_mel_spectrogram(self):
        """
        Возвращает Mel-спектрограмму для текущего аудио
        в формате, совместимом с моделью:
        (batch=1, channel=1, n_mels, time_frames) -> np.float32
        """
        audio, sr = self.get_audio_segment()

        # Проверяем, что аудио не пустое (тишина)
        if np.max(np.abs(audio)) < 0.001:
            # Возвращаем нулевую спектрограмму (нужного размера)
            # Оцениваем число временных фреймов
            # Длина сигнала = n_samples, hop_length = Config.HOP_LENGTH
            n_frames = int(self.target_samples / Config.HOP_LENGTH) + 1
            mel_spec = np.zeros((Config.N_MELS, n_frames), dtype=np.float32)
        else:
            # Используем единый преобразователь из data_loader
            mel_spec = self.transform.to_mel_spectrogram(audio)   # (n_mels, time)

        # Добавляем размерности: (1, 1, n_mels, time)
        mel_spec = np.expand_dims(mel_spec, axis=0)  # (1, n_mels, time)
        mel_spec = np.expand_dims(mel_spec, axis=0)  # (1, 1, n_mels, time)

        return mel_spec.astype(np.float32)

    def stop(self):
        """Останавливает поток."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("[AUDIO] Поток остановлен")


class ContinuousDetector:
    """
    Непрерывный детектор, объединяющий захват аудио и модель.
    """

    def __init__(self, model, capture: AudioCapture = None,
                 threshold: float = Config.THREAT_CONFIDENCE_THRESHOLD,
                 interval: float = 1.0):
        """
        Args:
            model: Обученная модель PyTorch (в режиме eval)
            capture: Экземпляр AudioCapture (если None, создаётся новый)
            threshold: Порог уверенности для объявления угрозы
            interval: Интервал между анализами (сек)
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.threshold = threshold
        self.interval = interval

        self.capture = capture or AudioCapture()
        self._running = False

    def start(self):
        """Запускает непрерывное обнаружение."""
        self.capture.start_stream()
        self._running = True

        print("\n[СИСТЕМА] Начинается непрерывный мониторинг...")
        print("[СИСТЕМА] Нажмите Ctrl+C для остановки\n")

        try:
            with torch.no_grad():
                while self._running:
                    # Получаем спектрограмму (numpy)
                    spectrogram = self.capture.get_mel_spectrogram()

                    # Конвертируем в тензор и передаём на устройство
                    tensor_input = torch.tensor(spectrogram, device=self.device)

                    # Инференс
                    output = self.model(tensor_input)
                    probs = torch.softmax(output, dim=1).cpu().numpy()

                    safe_prob = float(probs[0][0])
                    threat_prob = float(probs[0][1])

                    is_threat = threat_prob >= self.threshold

                    # Вывод
                    if is_threat:
                        print(f"⚠️  [УГРОЗА] Дрон обнаружен! Уверенность: {threat_prob:.1%}")
                    else:
                        print(f"✅ [БЕЗОПАСНО] Шумов нет. Уверенность: {safe_prob:.1%}")

                    time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n[СИСТЕМА] Остановка мониторинга...")
        finally:
            self.stop()

    def stop(self):
        """Останавливает мониторинг."""
        self._running = False
        self.capture.stop()


# ----------------------------------------------------------------------
# Точка входа для запуска непрерывного детектора из командной строки
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from src.model import get_model
    from src.inference import ThreatDetector

    parser = argparse.ArgumentParser(description="Непрерывный акустический мониторинг")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к файлу модели (.pth)")
    parser.add_argument("--threshold", type=float, default=Config.THREAT_CONFIDENCE_THRESHOLD,
                        help="Порог уверенности")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Интервал между анализами (сек)")
    parser.add_argument("--device", type=int, default=None,
                        help="Индекс звукового устройства")
    args = parser.parse_args()

    # Загружаем модель
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Config.MODEL_DIR / "best_model.pth"

    if not model_path.exists():
        print(f"[ОШИБКА] Модель не найдена: {model_path}")
        print("Сначала обучите модель: python main.py --train")
        sys.exit(1)

    print(f"[INFO] Загрузка модели из {model_path}")
    checkpoint = torch.load(model_path, map_location=Config.get_device(), weights_only=False)

    # Инициализируем архитектуру и загружаем веса
    model = get_model(Config.MODEL_TYPE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(Config.get_device())

    # Создаём захват
    capture = AudioCapture(device_index=args.device)

    # Запускаем детектор
    detector = ContinuousDetector(
        model=model,
        capture=capture,
        threshold=args.threshold,
        interval=args.interval
    )
    detector.start()
