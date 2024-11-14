import numpy as np
import pandas as pd

from cadv_exploration.error_injection.corrupts import Scaling


def test_scaling():
    df = pd.DataFrame({
        'a': np.arange(10),
        'b': np.arange(10),
        'c': np.arange(10),
    })
    scaling = Scaling(columns='a')
    corrupted_df = scaling.transform(df)
    assert df.shape == corrupted_df.shape
    assert df['a'].sum() < corrupted_df['a'].sum()
    assert df['b'].sum() == corrupted_df['b'].sum()
    assert df['c'].sum() == corrupted_df['c'].sum()
