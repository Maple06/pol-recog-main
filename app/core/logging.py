import logging, logging.config
import pytz
from pytz import timezone
from datetime import datetime

tz = pytz.timezone('Asia/Jakarta')
logging.config.fileConfig('logging.ini')
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Jakarta')).timetuple()
logger = logging.getLogger('app')