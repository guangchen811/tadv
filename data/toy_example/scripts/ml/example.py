class ColumnDetectionTask:

    def code_type(self):
        return "python"

    @property
    def original_script(self):
        return """
df = pd.read_csv("s3://datalake/latest/hospitalisations.csv")
df['cost_smoothed'] = np.log(df['cost'])
df['admission_day'].fillna(df['discharge_day'])
df['duration'] = df['discharge_day'] - df['admission_day']
categorical_cols = ['diagnosis', 'insurance']
for col in categorical_cols:
df[col] = pd.get_dummies(df[col], dummy_na=True)
features = df[categorical_cols + ['duration', 'cost_smoothed']]
labels = label_binarize(df['complications'], classes=['Y', 'N'])
model = sklearn.tree.DecisionTreeClassifier()
model.fit(train_features, train_labels)
deploy_to_production(model)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ["diagnosis", "admission_day", "discharge_day", "insurance", "cost", "complications"]
