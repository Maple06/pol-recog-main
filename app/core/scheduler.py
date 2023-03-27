# Run daily dataset update and encode

from ..api.api_v1.services.facerecog_service import recogService as RSv1
from ..api.api_v2.services.facerecog_service import recogService as RSv2
from ..api.api_v3.services.facerecog_service import recogService as RSv3
from apscheduler.schedulers.background import BackgroundScheduler
from .logging import logger

def dailyEncodev1():
    try:
        logger.info("[v1] Running daily dataset update and encoding...")
        RSv1.encodeFaces()
        logger.info("[v1] Daily dataset updated and encoded!")
    except:
        logger.error("[v1] Daily dataset update and encoding failed.")

def dailyEncodev2():
    try:
        logger.info("[v2v3] Running daily dataset update and encoding...")
        RSv2.encodeFaces()
        logger.info("[v2v3] Daily dataset updated and encoded!")
    except:
        logger.error("[v2v3] Daily dataset update and encoding failed.")

def dailyEncodev3():
    try:
        logger.info("[v2v3] Running daily dataset update and encoding...")
        RSv3.encodeFaces()
        logger.info("[v2v3] Daily dataset updated and encoded!")
    except:
        logger.error("[v2v3] Daily dataset update and encoding failed.")

scheduler = BackgroundScheduler()
scheduler.start()

# Run the daily update every day at 01:00 a.m.
scheduler.add_job(dailyEncodev1, 'cron', day_of_week='mon-sun', hour=1, minute=00)
scheduler.add_job(dailyEncodev2, 'cron', day_of_week='mon-sun', hour=1, minute=00)
scheduler.add_job(dailyEncodev3, 'cron', day_of_week='mon-sun', hour=1, minute=00)