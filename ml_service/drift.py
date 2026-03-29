import asyncio
import logging
import threading

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

from ml_service import config

logger = logging.getLogger(__name__)


class DriftMonitor:
    def __init__(self, batch_size: int, interval_sec: int, url: str, project_id: str):
        self.batch_size = batch_size
        self.interval_sec = interval_sec
        self.url = url
        self.project_id = project_id

        self._lock = threading.Lock()
        self._reference_rows: list[dict] = []
        self._current_rows: list[dict] = []
        self._reference_df: pd.DataFrame | None = None

    def add_event(self, row: dict) -> None:
        with self._lock:
            if self._reference_df is None:
                self._reference_rows.append(row)
            else:
                self._current_rows.append(row)

    def _rotate_batches(self) -> pd.DataFrame | None:
        with self._lock:
            if self._reference_df is None and len(self._reference_rows) >= self.batch_size:
                self._reference_df = pd.DataFrame(self._reference_rows[: self.batch_size])
                self._reference_rows = self._reference_rows[self.batch_size:]

            if self._reference_df is None or len(self._current_rows) < self.batch_size:
                return None

            current_df = pd.DataFrame(self._current_rows[: self.batch_size])
            self._current_rows = self._current_rows[self.batch_size:]
            return current_df

    async def run_forever(self) -> None:
        workspace = RemoteWorkspace(self.url)

        while True:
            await asyncio.sleep(self.interval_sec)
            current_df = self._rotate_batches()
            if current_df is None:
                continue

            try:
                report = Report(metrics=[DataDriftPreset()])
                result = report.run(
                    reference_data=self._reference_df,
                    current_data=current_df,
                )
                workspace.add_run(self.project_id, result)
            except Exception:
                logger.exception('Failed to build or upload evidently drift report')


drift_monitor = DriftMonitor(
    batch_size=int(config.getenv_default('DRIFT_BATCH_SIZE', '100')),
    interval_sec=int(config.getenv_default('DRIFT_INTERVAL_SEC', '300')),
    url=config.getenv_default('EVIDENTLY_URL', 'http://158.160.2.37:8000/'),
    project_id=config.getenv_default(
        'EVIDENTLY_PROJECT_ID',
        '019d061f-cc08-7b5e-b932-d792a1f258e2',
    ),
)