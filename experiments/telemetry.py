# experiments/telemetry.py
# GPU 전원/성능 계측 모듈.
# - PyNVML 우선 사용, 실패 시 nvidia-smi CLI 파싱으로 폴백.
# - PyNVML·nvidia-smi 모두 불가(Colab 환경 등) 시 torch.cuda 기반 TDP 추정 폴백
#   (PowerMeter.estimate) — 실측이 아닌 추정임을 호출부에서 Source 주석으로 명시.
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

# GPU TDP 상수표 (torch.cuda 기반 전력 추정 폴백용).
# Colab 무료 등급(T4) 포함 주요 GPU 의 전형적 TDP(W).
TDP_TABLE = {
    "Tesla T4": 70.0, "T4": 70.0,
    "Tesla P100": 250.0, "Tesla V100": 300.0, "Tesla K80": 300.0,
    "A100": 400.0, "A10G": 150.0, "L4": 72.0,
    "RTX 4090": 450.0, "RTX 4080": 320.0, "RTX 3090": 350.0,
    "RTX 3080": 320.0, "RTX 3060": 170.0, "RTX 2080 Ti": 250.0,
    "RTX 2080": 215.0, "RTX 2070": 175.0, "RTX 2060": 160.0,
    "GTX 1080 Ti": 250.0,
}


def _tdp_from_name(name):
    """GPU 이름 문자열에서 TDP 후보 탐색. 미지정 시 None."""
    for key, w in TDP_TABLE.items():
        if key in name:
            return w
    return None


class PowerMeter:
    """GPU 전원 계측기.

    measure(): (avg_power_W, total_energy_J) 반환.
    - PyNVML: powerDraw(W)를 주기 샘플링하여 사다리꼴 적분.
    - nvidia-smi 폴백: --query-gpu=power.draw --format=csv,noheader,nounits
    - 계측 불가 환경: (None, None) 반환하며 calling code가 에너지 모델을 쓰도록 함.
    estimate(): torch.cuda 기반 TDP 추정 폴백 (실계측 불가 시 호출).
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

    def estimate(self, elapsed_seconds, tdp=None):
        """torch.cuda 기반 전력/에너지 추정 (실계측 불가 환경의 최종 폴백).

        Colab 등에서 pynvml 초기화 실패 + nvidia-smi 미가용일 때,
        torch.cuda 로 GPU 이름을 얻어 TDP 상수표에서 전형 전력(W)을 찾고,
        측정 구간 전체를 busy 로 가정해 E[J] = P[W] × t[s] 로 추정한다.

        반환: (est_power_W, est_energy_J). torch.cuda 미가용 시 (None, None).
        """
        try:
            import torch
        except Exception:
            return None, None
        if not torch.cuda.is_available():
            return None, None
        name = torch.cuda.get_device_name(0)
        power = tdp or _tdp_from_name(name)
        if power is None:
            # 미식별 GPU: T4급(70W) 기본 가정 (Colab 무료 등급의 전형치)
            power = 70.0
        return float(power), float(power * max(0.0, elapsed_seconds))

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