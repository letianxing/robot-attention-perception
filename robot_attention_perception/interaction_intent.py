"""Explicit addressee detection shared by live and final ASR decisions."""
import re

def directed_call(text):
    return bool(re.match(r"^\s*(?:(?:你好|喂|嘿|hey|hi)[，,！!\s]*)?(?:reachy\b|机器人|小圆|小园|小元|小袁)",text,re.I))
