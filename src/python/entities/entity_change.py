from datetime import datetime
from typing import Tuple, List


class EntityChange:
    ID = "EntityId"
    ADDED_LINES = "AddedLines"
    DELETED_LINES = "DeletedLines"
    CONTRIBUTOR = "Contributor"
    BUG_FIX = "BugFix"
    COMMIT = "Commit"
    COMMIT_DATE = "CommitDate"
    MERGE_COMMIT = "MergeCommit"

    def __init__(
        self,
        id: int,
        added_lines: int,
        deleted_lines: int,
        contributor: int,
        bug_fix: bool,
        commit_hash: str,
        commit_date: datetime,
        merge_commit: bool,
    ):
        self.id = id
        self.added_lines = added_lines
        self.deleted_lines = deleted_lines
        self.contributor = contributor
        self.bug_fix = bug_fix
        self.commit_hash = commit_hash
        self.commit_date = commit_date
        self.merge_commit = merge_commit

    def to_dict(self):
        d = {
            EntityChange.ID: self.id,
            EntityChange.ADDED_LINES: self.added_lines,
            EntityChange.DELETED_LINES: self.deleted_lines,
            EntityChange.CONTRIBUTOR: self.contributor,
            EntityChange.BUG_FIX: self.bug_fix,
            EntityChange.COMMIT: self.commit_hash,
            EntityChange.COMMIT_DATE: self.commit_date,
            EntityChange.MERGE_COMMIT: self.merge_commit,
        }
        return d


class Contributor:
    ID = "Id"
    KEY = "Key"
    NAME = "Name"
    EMAIL = "Email"

    def __init__(self, id: int, key: str, name: str, email: str):
        self.id = id
        self.key = key
        self.name = name
        self.email = email

    def to_dict(self):
        d = {
            Contributor.ID: self.id,
            Contributor.KEY: self.key,
            Contributor.NAME: self.name,
            Contributor.EMAIL: self.email,
        }
        return d

    @staticmethod
    def from_dict(d):
        return Contributor(
            d[Contributor.ID],
            d[Contributor.KEY],
            d[Contributor.NAME],
            d[Contributor.EMAIL],
        )
