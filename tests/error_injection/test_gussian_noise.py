import numpy as np
import pandas as pd

from tadv.error_injection.corrupts import GaussianNoise


def test_gaussian_noise():
    df = pd.DataFrame({
        'a': np.arange(10),
        'b': np.arange(10),
        'c': np.arange(10),
    })
    noise = GaussianNoise(columns='a')
    corrupted_df = noise.transform(df)
    assert df.shape == corrupted_df.shape
    assert df['a'].sum() != corrupted_df['a'].sum()
    assert df['b'].sum() == corrupted_df['b'].sum()
    assert df['c'].sum() == corrupted_df['c'].sum()

    noise_multiple_columns = GaussianNoise(columns=['a', 'b'])
    corrupted_df = noise_multiple_columns.transform(df)
    assert df.shape == corrupted_df.shape
    assert df['a'].sum() != corrupted_df['a'].sum()
    assert df['b'].sum() != corrupted_df['b'].sum()
    assert df['c'].sum() == corrupted_df['c'].sum()
