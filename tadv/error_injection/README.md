#### Error Injection

The error injection module is built based on [Jenga](https://github.com/schelterlabs/jenga), a library for injecting
errors into datasets. We extend the error injection methods into more real world scenarios where we often need
context information to fix the errors. You can find the error injection
methods [here](/tadv/error_injection/corrupts).

The following table lists the error injection methods we support:

| **Type**                                                                                 | **Explanation**                                                                                               |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [Missing categorical value](/tadv/error_injection/corrupts/categorical_value_missing.py) | Replace one or more types of categorical value with a missing value or other existing values, or delete them. |
| [Dropping column](/tadv/error_injection/corrupts/column_dropping.py)                     | Drop one or more columns.                                                                                     |
| [Inserting column](/tadv/error_injection/corrupts/column_inserting.py)                   | Insert one or more columns by copying existing columns or generating new columns.                             |
| [Adding gaussian noise](/tadv/error_injection/corrupts/gaussian_noise.py)                | Add Gaussian noise to numerical values.                                                                       |
| [Masking values](/tadv/error_injection/corrupts/masking_values.py)                       | Mask one or more values in the dataset.                                                                       |
| [Scaling values](/tadv/error_injection/corrupts/scaling_values.py)                       | Scale numerical values by a factor.                                                                           |
