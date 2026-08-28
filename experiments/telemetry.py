# experiments/telemetry.py
# GPU 전원/성능 계측 모듈.
# - PyNVML 우선 사용, 실패 시 nvidia-smi 파싱으로 폴백.
# - CPU 전용 폴백(전원 계측 불가) 경고와 함께 에너지 모델 추정도 제공.
import time
import os
import subprocess
import threading

try:
    import pynvml
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


class PowerMeter:
    """GPU 전원 계측기.

    measure(): (avg_power_W, total_energy_J) 반환.
    - PyNVML: powerDraw(W)를 주기 샘플링하여 사다리꼴 적분.
    - nvidia-smi 폴백: --query-gpu=power.draw --format=csv,noheader,nounits
    - 계측 불가 환경: (None, None) 반환하며 calling code가 에너지 모델을 쓰도록 함.
    """

    def __init__(self, sample_interval=0.05):
        self.sample_interval = sample_interval
        self._nvml_handle = None
        self._use_pynvml = False
        self._use_smi = False
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                self._use_pynvml = True
            except Exception:
                self._use_pynvml = False
        if not self._use_pynvml:
            if self._smi_power_read() is not None:
                self._use_smi = True

    def _smi_power_read(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL)
            return float(out.decode().strip().split("\n")[0])
        except Exception:
            return None

    def _instant_watt(self):
        if self._use_pynvml:
            return pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0  # mW->W
        if self._use_smi:
            v = self._smi_power_read()
            if v is not None:
                return v
        return None

    @property
    def available(self):
        return self._use_pynvml or self._use_smi

    def measure(self, fn, *args, **kwargs):
        """fn 실행 중 전력을 샘플링하여 (avg_power_W, total_energy_J) 반환."""
        if not self.available:
            return None, None
        samples = []

        def sampler():
            while not stop_event.is_set():
                w = self._instant_watt()
                if w is not None:
                    samples.append((time.time(), w))
                stop_event.wait(self.sample_interval)

        stop_event = threading.Event()
        t = threading.Thread(target=sampler, daemon=True)
        t0 = time.time()
        t.start()
        try:
            result = fn(*args, **kwargs)
        finally:
            stop_event.set()
            t.join()
            t1 = time.time()
        if len(samples) < 2:
            return None, None
        # 사다리꼴 적분으로 에너지(J) 계산
        samples.sort(key=lambda x: x[0])
        ts = [s[0] for s in samples]
        ws = [s[1] for s in samples]
        # 정적 전력 상당량 제거를 위한 최소 샘플(기준)은 analysis 쪽에서 처리
        energy = 0.0
        for i in range(1, len(samples)):
            energy += (ws[i - 1] + ws[i]) / 2.0 * (ts[i] - ts[i - 1])
        avg = sum(ws) / len(ws)
        return avg, energy


def estimate_energy_from_avg(avg_power_W, seconds):
    """avg_power * seconds 근사(J). (계측 불가 시 에너지 모델 폴백)"""
    return avg_power_W * seconds


if __name__ == "__main__":
    m = PowerMeter()
    print(f"PowerMeter available={m.available} (pynvml={m._use_pynvml}, smi={m._use_smi})")