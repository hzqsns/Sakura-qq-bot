class NoiseFilter:
    """Decides whether a group message should be excluded from context recording."""

    @staticmethod
    def should_filter(
        text: str,
        has_image: bool,
        is_command: bool,
        min_length: int = 3,
    ) -> bool:
        """
        Return True if the message should NOT be recorded to context.

        Rules:
        - Commands (starting with /) are always filtered.
        - Messages with images are always kept, regardless of text length.
        - Pure text messages shorter than min_length are filtered.
        """
        if is_command:
            return True
        if has_image:
            return False
        return len((text or "").strip()) < min_length
