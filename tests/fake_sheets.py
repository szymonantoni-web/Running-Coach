"""An in-memory stand-in for the slice of gspread that SheetsStore uses.

SheetsStore only ever calls five methods on a worksheet — `get_all_values`,
`clear`, `append_rows`, plus `worksheet`/`add_worksheet` on the spreadsheet.
Reimplementing those over a list of lists means the whole Sheets backend can be
tested without credentials, a network, or a Google account, and the tests run in
milliseconds.

It deliberately reproduces two behaviours that cause real bugs:

* every cell is a string, because that is what the Sheets API returns even for
  a number you wrote as a number;
* `get_all_values` trims nothing, so trailing blank rows look exactly as
  awkward as they do in practice.
"""

from __future__ import annotations


class WorksheetNotFound(Exception):
    """gspread raises its own class; the store catches broadly, so any
    exception type works here."""


class FakeWorksheet:
    def __init__(self, title: str, rows: int = 1000, cols: int = 26):
        self.title = title
        self.rows = rows
        self.cols = cols
        self._values: list[list[str]] = []
        self.append_calls = 0
        self.clear_calls = 0

    def get_all_values(self) -> list[list[str]]:
        return [list(row) for row in self._values]

    def clear(self) -> None:
        self.clear_calls += 1
        self._values = []

    def append_rows(self, rows, value_input_option: str = "RAW") -> None:
        self.append_calls += 1
        for row in rows:
            # The API stringifies everything on the way back out.
            self._values.append([("" if cell is None else str(cell)) for cell in row])

    # -- helpers for the tests themselves -----------------------------------
    def seed(self, rows) -> None:
        self._values = [[("" if c is None else str(c)) for c in row] for row in rows]


class FakeSpreadsheet:
    def __init__(self):
        self._worksheets: dict[str, FakeWorksheet] = {}

    def worksheet(self, title: str) -> FakeWorksheet:
        if title not in self._worksheets:
            raise WorksheetNotFound(title)
        return self._worksheets[title]

    def add_worksheet(self, title: str, rows: int = 1000, cols: int = 26) -> FakeWorksheet:
        sheet = FakeWorksheet(title, rows, cols)
        self._worksheets[title] = sheet
        return sheet

    # -- helpers ------------------------------------------------------------
    @property
    def titles(self) -> list[str]:
        return sorted(self._worksheets)

    def raw(self, title: str) -> list[list[str]]:
        return self._worksheets[title].get_all_values()
