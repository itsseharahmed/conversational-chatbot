import re


CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[ .,-]{0,2}){12,18}\d(?!\d)")
SECURITY_CODE = re.compile(
    r"\b(?P<label>cvv2?|cvc2?|security code|otp"
    r"|(?:verification|authentication|auth|login|reset|recovery|access|sms|2fa|mfa|one[- ]time)\s+(?:code|password|passcode)"
    r"|pin(?:\s+(?:number|code))?)\b"
    r"(?P<sep>\W{0,5}(?:is|was)?\W{0,5})(?P<code>(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{2,15})\b",
    re.IGNORECASE,
)
PASSWORD = re.compile(
    r"\b(password|passcode|passphrase|pwd)\b(\s*(?:(?:is|was)\b\s*[:=]?|[:=])\s*)"
    r"(?!\[|(?:not|wrong|incorrect|invalid|expired|working|being|correct|reset|changed|updated|rejected|accepted)\b)"
    r"(?:\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
REMOVED_PLACEHOLDER = re.compile(r"\[(?:card number|code|password) removed\]")

SAFETY_WORDS = re.compile(
    r"\b(?:burn\w*|smok\w*|spark\w*|overheat\w*|melt\w*|fire|flames?|explode\w*|fumes?|electrocute\w*|shock\w*|leak\w*)\b",
    re.IGNORECASE,
)

SAFETY_NOTICE = (
    "If your product is smoking, sparking, overheating, leaking, burning or smells burnt, stop using it now and "
    "disconnect power only if it is safe. If there is fire, heavy smoke or immediate danger, move away and "
    "call local emergency services."
)
SENSITIVE_WARNING = (
    "Please don't share passwords, one-time codes, recovery codes, full card numbers or card security codes "
    "in chat. Northstar Home never asks for them, and what you shared has not been used or saved."
)
SENSITIVE_HELP = (
    "If you are having trouble signing in or with a payment, describe the problem without those details, "
    "for example the error message you see."
)
SENSITIVE_ONLY_REPLY = f"{SENSITIVE_WARNING} {SENSITIVE_HELP}"


def mask_sensitive(text):
    text = CARD_NUMBER.sub("[card number removed]", text)
    text = SECURITY_CODE.sub(
        lambda match: f"{match.group('label')}{match.group('sep')}[code removed]",
        text,
    )
    return PASSWORD.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[password removed]",
        text,
    )
