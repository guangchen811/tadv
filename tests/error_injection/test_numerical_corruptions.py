import numpy as np
import pandas as pd


# def test_gaussian_noise():
#     df = pd.DataFrame({
#         'a': np.arange(10),
#         'b': np.arange(10),
#         'c': np.arange(10),
#     })
#     gaussian_noise = GaussianNoise(fraction=0.1, column='a')
#     corrupted_df = gaussian_noise.transform(df)
#     assert df.shape == corrupted_df.shape
#     assert not df.equals(corrupted_df)
