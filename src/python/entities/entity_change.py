class EntityChange:
    ID = "EntityId"
    ADDED_LINES = "AddedLines"
    DELETED_LINES = "DeletedLines"
    CONTRIBUTOR = "Contributor"
    COMMIT = "Commit"

    def __init__(
        self,
        id: int,
        added_lines: int,
        deleted_lines: int,
        contributor: int,
        commit_hash: str,
    ):
        self.id = id
        self.added_lines = added_lines
        self.deleted_lines = deleted_lines
        self.contributor = contributor
        self.commit_hash = commit_hash

    def to_dict(self):
        d = {
            EntityChange.ID: self.id,
            EntityChange.ADDED_LINES: self.added_lines,
            EntityChange.DELETED_LINES: self.deleted_lines,
            EntityChange.CONTRIBUTOR: self.contributor,
            EntityChange.COMMIT: self.commit_hash,
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
