from datetime import datetime
from typing import Tuple, List


class EntityChange:
    ID = "EntityId"
    ADDED_LINES = "AddedLines"
    DELETED_LINES = "DeletedLines"
    DMM_SIZE_LR = "DMMSizeLowRisk"
    DMM_SIZE_HR = "DMMSizeHighRisk"
    DMM_COMPLEXITY_LR = "DMMComplexityLowRisk"
    DMM_COMPLEXITY_HR = "DMMComplexityHighRisk"
    DMM_INTERFACING_LR = "DMMInterfacingLowRisk"
    DMM_INTERFACING_HR = "DMMInterfacingHighRisk"
    ADDED_CHANGE_SCATTERING = "AddedChangeScattering"
    DELETED_CHANGE_SCATTERING = "DeletedChangeScattering"
    CONTRIBUTOR = "Contributor"
    BUG_FIX = "BugFix"
    COMMIT = "Commit"
    COMMIT_DATE = "CommitDate"
    MERGE_COMMITS = "MergeCommits"

    def __init__(
        self,
        id: int,
        added_lines: int,
        deleted_lines: int,
        change_scattering: Tuple[int, int],
        dmm_size: Tuple[int, int],
        dmm_complexity: Tuple[int, int],
        dmm_interfacing: Tuple[int, int],
        contributor: int,
        bug_fix: bool,
        commit_hash: str,
        commit_date: datetime,
    ):
        self.id = id
        self.added_lines = added_lines
        self.deleted_lines = deleted_lines
        self.added_change_scattering = change_scattering[0]
        self.deleted_change_scattering = change_scattering[1]
        self.dmm_size_lr = dmm_size[0]
        self.dmm_size_hr = dmm_size[1]
        self.dmm_complexity_lr = dmm_complexity[0]
        self.dmm_complexity_hr = dmm_complexity[1]
        self.dmm_interfacing_lr = dmm_interfacing[0]
        self.dmm_interfacing_hr = dmm_interfacing[1]
        self.contributor = contributor
        self.bug_fix = bug_fix
        self.commit_hash = commit_hash
        self.commit_date = commit_date
        self.merge_commits = []

    def to_dict(self):
        d = {
            EntityChange.ID: self.id,
            EntityChange.ADDED_LINES: self.added_lines,
            EntityChange.DELETED_LINES: self.deleted_lines,
            EntityChange.ADDED_CHANGE_SCATTERING: self.added_change_scattering,
            EntityChange.DELETED_CHANGE_SCATTERING: self.deleted_change_scattering,
            EntityChange.DMM_SIZE_LR: self.dmm_size_lr,
            EntityChange.DMM_SIZE_HR: self.dmm_size_hr,
            EntityChange.DMM_COMPLEXITY_LR: self.dmm_complexity_lr,
            EntityChange.DMM_COMPLEXITY_HR: self.dmm_complexity_hr,
            EntityChange.DMM_INTERFACING_LR: self.dmm_interfacing_lr,
            EntityChange.DMM_INTERFACING_HR: self.dmm_interfacing_hr,
            EntityChange.CONTRIBUTOR: self.contributor,
            EntityChange.BUG_FIX: self.bug_fix,
            EntityChange.COMMIT: self.commit_hash,
            EntityChange.COMMIT_DATE: self.commit_date,
            EntityChange.MERGE_COMMITS: self.merge_commits,
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
