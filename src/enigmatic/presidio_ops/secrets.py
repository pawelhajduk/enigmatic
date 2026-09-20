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
                    "pem_begin",
                    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
                    0.95,
                )
            ],
        ),
    ]
