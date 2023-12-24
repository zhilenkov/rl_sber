"""
    Project structure with all kinds of path variables.
"""

from pathlib import Path


class Config:
    def __init__(self):

        self.project = Path(__file__).parent.parent.parent
        self.docs = self.project / "docs"
        self.reports = self.project / "reports"
        self.reports_included = self.reports / "included"
        self.reports_excluded = self.reports / "excluded"
        self.models = self.project / "models"
        self.package = self.project / "sberl"
