from typing import Literal

NULL_KEY = object()

DEFAULT_ALPHA = 0.05
DEFAULT_N_RESAMPLES = 10_000

# inference strategy options
InferenceMethod = Literal["analytical", "bootstrap"]
DEFAULT_INFERENCE_METHOD: InferenceMethod = "analytical"
