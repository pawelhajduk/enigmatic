"""Custom Presidio PatternRecognizers for secrets common in coding prompts."""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


def secret_recognizers() -> list[PatternRecognizer]:
    """GitHub, OpenAI, AWS, and PEM private-key recognizers."""
    return [
        PatternRecognizer(
            supported_entity="GITHUB_TOKEN",
            name="GithubTokenRecognizer",
            patterns=[
                Pattern("github_pat", r"\bghp_[A-Za-z0-9]{36}\b", 0.9),
                Pattern("github_finegrained", r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", 0.9),
                Pattern("github_oauth", r"\bgho_[A-Za-z0-9]{36}\b", 0.85),
            ],
        ),
        PatternRecognizer(
            supported_entity="OPENAI_KEY",
            name="OpenAIKeyRecognizer",
            patterns=[
                Pattern("openai_sk", r"\bsk-[A-Za-z0-9]{20,}\b", 0.85),
                Pattern("openai_proj", r"\bsk-proj-[A-Za-z0-9_-]{20,}\b", 0.9),
            ],
        ),
        PatternRecognizer(
            supported_entity="AWS_ACCESS_KEY",
            name="AwsAccessKeyRecognizer",
            patterns=[Pattern("aws_akia", r"\bAKIA[0-9A-Z]{16}\b", 0.9)],
        ),
        PatternRecognizer(
            supported_entity="AWS_SECRET_KEY",
            name="AwsSecretKeyRecognizer",
            patterns=[
                Pattern(
                    "aws_secret",
                    r"(?i)aws_secret_access_key\s*[=:]\s*([A-Za-z0-9/+=]{40})",
                    0.8,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="PEM_KEY",
            name="PemKeyRecognizer",
            patterns=[
                Pattern(
                    "pem_block",
                    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
                    r"[\s\S]{0,16000}?"
                    # \Z, not $: Presidio compiles with re.MULTILINE, where $ ends the BEGIN line.
                    r"(?:-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----|\Z)",
                    0.95,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="JWT",
            name="JwtRecognizer",
            patterns=[
                Pattern(
                    "jwt",
                    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b",
                    0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="SLACK_TOKEN",
            name="SlackTokenRecognizer",
            patterns=[Pattern("slack", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", 0.9)],
        ),
        PatternRecognizer(
            supported_entity="STRIPE_KEY",
            name="StripeKeyRecognizer",
            patterns=[
                Pattern("stripe", r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b", 0.9)
            ],
        ),
        PatternRecognizer(
            supported_entity="CONNECTION_STRING",
            name="ConnectionStringRecognizer",
            patterns=[
                Pattern(
                    "db_url",
                    r"\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s'\"<>]+",
                    0.85,
                )
            ],
        ),
    ]
