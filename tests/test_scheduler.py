from job_radar.ingest.scheduler import build_scheduler


def test_build_scheduler_registers_daily_job_with_safety_settings() -> None:
    scheduler = build_scheduler()
    job = scheduler.get_job("daily_ingestion")

    assert job is not None
    assert str(job.trigger) == "cron[hour='3', minute='0']"
    assert job.coalesce is True
    assert job.max_instances == 1
    assert job.misfire_grace_time == 10800
