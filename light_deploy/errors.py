class DecoderBundleError(ValueError):
    pass


class ContextLengthError(ValueError):
    def __init__(self, total_tokens: int, max_model_len: int) -> None:
        super().__init__(f"Prompt and generation length {total_tokens} exceeds max_model_len {max_model_len}")
