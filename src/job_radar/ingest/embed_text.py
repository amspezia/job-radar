def build_embed_text(
    title: str,
    description: str,
    requirements: str | None,
    responsibilities: str | None,
) -> str:
    """Build the text a posting is embedded from.

    Kept free of job_radar imports so the embedding eval harness can rebuild the
    exact production text without loading job_radar.config.
    """
    # Embed role-specific content only; fall back to full description
    # when extraction produced nothing (LLM failure or very short posting).
    if requirements or responsibilities:
        return "\n".join(filter(None, [title, requirements, responsibilities]))
    return f"{title}\n{description}"
