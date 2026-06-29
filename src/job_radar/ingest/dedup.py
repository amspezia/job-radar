import hashlib

from job_radar.adapters.sources.base import NormalizedJob


def content_hash(job: NormalizedJob) -> str:
    """Stable identity hash for cross-source dedup: company + title + location."""
    location = "none" if not job.location else job.location
    normalized_fields = [" ".join(s.split()).lower() for s in [job.company, job.title, location]]
    return hashlib.sha256("|".join(normalized_fields).encode("utf-8")).hexdigest()
