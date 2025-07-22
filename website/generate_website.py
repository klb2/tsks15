import os
import re
from datetime import date, datetime, timedelta
import itertools

import jinja2
import yaml

from constants import (
    DIR_TEMPLATES,
    DIR_PUBLIC,
    DIR_DATA,
    DIR_NOTEBOOKS,
    FILE_SCHEDULE,
    FILE_INFO,
    FILE_LABS,
    MAP_SESSION_TYPE,
)

DIR_HERE = os.path.dirname(__file__)
RE_DURATION = r"PT(?:([0-9]{1,2})H)?(?:([0-9]{1,2})M)?"


def load_data(data_file):
    file_info = os.path.join(DIR_HERE, DIR_DATA, data_file)
    with open(file_info, "r") as yml_file:
        data = yaml.safe_load(yml_file)
    return data


def create_schedule():
    data = load_data(FILE_SCHEDULE)
    schedule = []
    keyfunc = lambda x: x.get("type", "Unknown")
    data = sorted(data, key=keyfunc)
    for k, g in itertools.groupby(data, keyfunc):
        _group = MAP_SESSION_TYPE.get(k, "Unknown")
        for num_session, session in enumerate(g):
            _start_time, _end_time = session["time"].split("/", 1)
            start_time = datetime.fromisoformat(_start_time)
            if _end_time.startswith("P"):
                _duration_hour, _duration_min = re.match(
                    RE_DURATION, _end_time
                ).groups()
                duration = timedelta(
                    hours=int(_duration_hour), minutes=int(_duration_min)
                )
                end_time = start_time + duration
            else:
                end_time = datetime.fromisoformat(
                    f"{start_time.date().isoformat()}T{_end_time}"
                )
            topics = session.get("topics", [])
            reading = session.get("reading")
            _session = {
                "title": f"{_group}&nbsp;{num_session + 1:d}",
                "start_time": start_time,
                "end_time": end_time,
                "topics": topics,
                "reading": reading,
                "type": k,
            }
            schedule.append(_session)
    return schedule


def get_all_notebooks():
    notebooks = {}
    dir_notebooks = os.path.normpath(os.path.join(DIR_HERE, DIR_NOTEBOOKS))
    dirs_lecture = [
        d
        for d in os.listdir(dir_notebooks)
        if os.path.isdir(os.path.join(dir_notebooks, d))
    ]
    for _dir in dirs_lecture:
        _dir_lecture = os.path.join(dir_notebooks, _dir)
        _notebooks = [
            f
            for f in os.listdir(_dir_lecture)
            if os.path.isfile(os.path.join(_dir_lecture, f)) and f.endswith(".py")
        ]
        notebooks[_dir] = _notebooks
    return notebooks


def main():
    now = datetime.now()
    dir_template = os.path.join(DIR_HERE, DIR_TEMPLATES)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(dir_template))

    general_info = load_data(FILE_INFO)
    course_info = general_info.pop("general_info")
    schedule = create_schedule()
    lab_info = load_data(FILE_LABS)

    notebook_info = get_all_notebooks()

    pages = {
        "index": ("index.html", "General Information", general_info),
        "schedule": ("schedule.html", "Schedule", schedule),
        "labs": ("labs.html", "Labs", lab_info),
        "notebooks": ("notebooks.html", "Notebooks", notebook_info),
    }

    for page, info in pages.items():
        _template, _page_name, _data = info
        template = env.get_template(_template)
        content = template.render(
            course_info=course_info,
            page_name=_page_name,
            now=now,
            data=_data,
        )
        out_path = os.path.join(DIR_HERE, DIR_PUBLIC, f"{page}.html")
        with open(out_path, "w") as html_file:
            html_file.write(content)


if __name__ == "__main__":
    results = main()
