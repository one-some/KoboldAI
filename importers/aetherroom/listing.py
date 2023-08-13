# Not production ready by any means. If this were to be used widely I would like
# to use a caching server to protect the club server as well as users.

import functools
import os
import shutil
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import List
import lorem

import requests

from logger import logger

DB_PATH = "data/club.db"
DB_URL = "https://aetherroom.club/backup.db"


@dataclass
class Prompt:
    id: int
    title: str
    desc: str
    content: str
    tags: List[str]
    nsfw: bool
    date_created: str


class Sort(Enum):
    ID_DESC = 0
    ID_ASC = 1
    RANDOM = 2


class ClubPromptDB:
    def __init__(self) -> None:
        assert not os.path.isdir(DB_PATH)

        # TODO: Update if too old
        if not os.path.exists(DB_PATH):
            logger.info("Prompt DB not downloaded. Downloading...")
            self.update_db()

        self.con = sqlite3.connect(DB_PATH)

    def update_db(self) -> None:
        if os.path.isfile(DB_PATH):
            os.remove(DB_PATH)

        with requests.get(DB_URL, stream=True) as r:
            # Handle different compression techniques, as per:
            # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests#comment95588469_39217788
            r.raw.read = functools.partial(r.raw.read, decode_content=True)

            with open(DB_PATH, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    def get_prompts(
        self, limit: int = 25, offset: int = 0, sort: Sort = Sort.ID_DESC
    ) -> Prompt:
        ret = []

        raw_sort_quantifier = {
            Sort.ID_DESC: "Id DESC",
            Sort.ID_ASC: "Id ASC",
            Sort.RANDOM: "RANDOM()",
        }[sort]

        for p_id, desc, is_nsfw, tags, title, date_created, content in self.con.execute(
            # Be mindful of f-string / raw string manipulation here :^)
            f"SELECT Id,Description,Nsfw,Tags,Title,DateCreated,PromptContent FROM Prompts ORDER BY {raw_sort_quantifier} LIMIT ? OFFSET ?;",
            (limit, offset),
        ):
            ret.append(
                # Prompt(
                #     p_id,
                #     title=title,
                #     desc=desc,
                #     content=content,
                #     tags=tags.split(", "),
                #     nsfw=bool(is_nsfw),
                #     date_created=date_created,
                # )
                Prompt(
                    p_id,
                    title=lorem.sentence().strip(".")[:35],
                    desc=lorem.paragraph(),
                    content=lorem.paragraph(),
                    tags=lorem.sentence().lower().strip(".").split(" "),
                    # tags=tags.split(", "),
                    nsfw=bool(is_nsfw),
                    date_created=date_created,
                )
            )
        return ret
