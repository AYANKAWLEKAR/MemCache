"""Celery application: broker and result backend from settings."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "memcache",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    task_soft_time_limit=540,
    task_default_retry_delay=30,
    task_acks_late=True,
)
