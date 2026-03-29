import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]

NUMERIC_COLUMNS = {
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
}


def to_dataframe(req: PredictRequest, needed_columns: list[str] | None = None) -> pd.DataFrame:
    columns = needed_columns or FEATURE_COLUMNS

    unknown = [column for column in columns if column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f'Unknown feature(s) requested by model: {unknown}')

    payload = req.model_dump(by_alias=True)

    missing = [column for column in columns if payload.get(column) is None]
    if missing:
        raise ValueError(f'Missing required feature(s): {missing}')

    row = {column: payload[column] for column in columns}
    return pd.DataFrame([row], columns=columns)