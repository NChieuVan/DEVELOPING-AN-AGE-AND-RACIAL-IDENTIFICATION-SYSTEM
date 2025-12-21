# write_log.py
import time
import threading
import os

class FPSStressLogger:
    """
    FPS logger với cửa sổ trượt (sliding window):
    - Luôn ghi FPS khi có face
    - Giữ tối đa N giây gần nhất (mặc định 5 phút)
    - Khi chương trình kết thúc -> luôn ghi file
    """

    def __init__(self, window_sec=300, log_file="stress_test_fps_log.txt"):
        self.window_sec = window_sec
        self.log_file = log_file

        self.records = []   # list of (timestamp, fps)
        self.lock = threading.Lock()

        # nếu file cũ tồn tại -> load lại
        self._load_existing()

    def _load_existing(self):
        if not os.path.exists(self.log_file):
            return

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("FPS:"):
                    parts = line.strip().split()
                    ts = float(parts[1])
                    fps = float(parts[2])
                    self.records.append((ts, fps))
        except Exception:
            # nếu file cũ lỗi format thì bỏ qua
            self.records = []

    def update(self, e2e_latency_sec: float, face_count: int):
        if face_count < 1 or e2e_latency_sec <= 0:
            return

        fps = 1.0 / e2e_latency_sec
        now = time.time()

        with self.lock:
            self.records.append((now, fps))
            self._trim_window()

    def _trim_window(self):
        """Giữ lại window_sec gần nhất"""
        now = time.time()
        cutoff = now - self.window_sec
        self.records = [(t, f) for (t, f) in self.records if t >= cutoff]

    def save(self):
        """Ghi log ra file (luôn gọi khi thoát chương trình)"""
        with self.lock:
            if not self.records:
                return

            fps_values = [f for (_, f) in self.records]
            avg_fps = sum(fps_values) / len(fps_values)
            min_fps = min(fps_values)
            max_fps = max(fps_values)

            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("=== STRESS TEST FPS LOG (SLIDING WINDOW) ===\n")
                f.write("Duration: 5 minutes\n")
                f.write(f"Window (sec): {self.window_sec}\n")
                f.write(f"Samples: {len(fps_values)}\n")
                f.write(f"Average FPS (E2E): {avg_fps:.2f}\n")
                f.write(f"Min FPS: {min_fps:.2f}\n")
                f.write(f"Max FPS: {max_fps:.2f}\n\n")

                f.write("---- RAW FPS RECORDS ----\n")
                for ts, fps in self.records:
                    f.write(f"FPS: {ts:.3f} {fps:.3f}\n")

            print(f"[FPS-LOGGER] Log saved ({len(fps_values)} samples) -> {self.log_file}")
