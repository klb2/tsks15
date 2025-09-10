import os.path
from zoneinfo import ZoneInfo

DIR_TEMPLATES = "templates"
DIR_PUBLIC = "public"
DIR_DATA = "data"
DIR_NOTEBOOKS = os.path.join(os.path.pardir, "notebooks")

FILE_INFO = "info.yml"
FILE_SCHEDULE = "schedule.yml"
FILE_NEWS = "news.yml"
FILE_LABS = "labs.yml"

MAP_SESSION_TYPE = {
    "lecture": "Lecture",
    "tutorial": "Tutorial",
    "lab": "Lab",
}

TIMEZONE = ZoneInfo("Europe/Stockholm")
