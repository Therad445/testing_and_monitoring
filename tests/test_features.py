import pytest

from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_uses_only_needed_columns():
    req = PredictRequest(age=25, **{'hours.per.week': 40})
    df = to_dataframe(req, needed_columns=['age', 'hours.per.week'])

    assert list(df.columns) == ['age', 'hours.per.week']
    assert df.iloc[0]['age'] == 25
    assert df.iloc[0]['hours.per.week'] == 40


def test_to_dataframe_raises_on_missing_feature():
    req = PredictRequest(age=25)

    with pytest.raises(ValueError, match='Missing required feature'):
        to_dataframe(req, needed_columns=['age', 'hours.per.week'])


def test_to_dataframe_raises_on_unknown_feature():
    req = PredictRequest(age=25)

    with pytest.raises(ValueError, match='Unknown feature'):
        to_dataframe(req, needed_columns=['age', 'unknown_feature'])