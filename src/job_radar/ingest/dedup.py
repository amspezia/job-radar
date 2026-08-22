import hashlib

from job_radar.adapters.sources.base import NormalizedJob


def content_hash(job: NormalizedJob) -> str:
    """Stable identity hash for cross-source dedup: company + title, plus
    location for non-remote postings only.

    Location is deliberately excluded for remote postings. Aggregators list one
    remote role once per eligible country (Mindrift, CapsLock, Bjak and
    Bluelight Consulting all do this; one real posting was found duplicated 27
    times), and with location in the hash every country is a separate identity,
    so none of those rows ever dedupe against each other. They then crowd the
    retrieval candidate pool and burn a fit-analysis LLM call each.

    For a non-remote posting the location IS part of the identity — "Software
    Engineer @ Google, NYC" and the same title in London are different roles.
    """
    fields = [job.company, job.title]
    if not job.remote:
        fields.append(job.location or "none")
    normalized_fields = [" ".join(s.split()).lower() for s in fields]
    return hashlib.sha256("|".join(normalized_fields).encode("utf-8")).hexdigest()
