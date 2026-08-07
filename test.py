"""Обратная совместимость: старые скрипты импортируют test.py."""
from asr import init_asr_models, run_segment_pipeline, unload_asr_models, run_test

__all__ = ["init_asr_models", "run_segment_pipeline", "unload_asr_models", "run_test"]

if __name__ == "__main__":
    run_test()
