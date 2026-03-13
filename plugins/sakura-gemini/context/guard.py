import re


# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(previous|prior|above|everything)",
    r"(你现在是|你是|assume you are|act as|pretend (to be|you are))\s*.{0,20}(没有限制|no\s*restrict|jailbreak|DAN|开发者模式|developer\s*mode)",
    r"(输出|告诉我|print|reveal|show|display)\s*.{0,20}(系统提示|system\s*prompt|指令|你的设定|你的配置|your\s*instructions?)",
    r"(忽略|忘记|override|bypass)\s*.{0,20}(之前|前面|上面|所有).{0,10}(指令|规则|限制|设定)",
    r"(开发者模式|developer\s*mode|jailbreak|越狱|DAN\b)",
    r"你(真正的|实际的|底层的)(身份|提示词|指令|系统)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class PromptGuard:
    @staticmethod
    def is_injection(text: str) -> bool:
        """Return True if the text looks like a prompt injection attempt."""
        if not text:
            return False
        for pattern in _COMPILED:
            if pattern.search(text):
                return True
        return False
