"""PEP 440 version parsing stub."""


class Version:
    def __init__(self, version):
        self.version = version

    def __str__(self):
        return self.version

    def __repr__(self):
        return f"Version('{self.version}')"
