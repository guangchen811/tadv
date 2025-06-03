import oyaml as yaml

from tadv.dq_manager import DeequDataQualityManager
from tadv.inspector.abstract_inspector_manager import AbstractInspectorManager


class DeequInspectorManager(AbstractInspectorManager):

    def spark_df_to_column_desc(self, spark_df, spark):
        result_dict = self.spark_df_to_column_desc_dict(spark, spark_df)
        yaml_string = yaml.dump(result_dict, default_flow_style=False, sort_keys=False)
        return yaml_string

    def spark_df_to_column_desc_dict(self, spark, spark_df):
        dq_manager = DeequDataQualityManager()
        result = dq_manager.profile_on_spark_df(spark, spark_df)
        column_names = list(result.profiles.keys())
        result_dict = {}
        for column_name in column_names:
            column_profile = result.profiles[column_name]
            column_profile_dict = self._profile_to_dict(column_profile)
            result_dict[column_name] = column_profile_dict
        return result_dict

    def deequ_result_desc(self, spark_df, spark):
        raise NotImplementedError("TODO")

    @staticmethod
    def _profile_to_dict(column_profile):
        if column_profile.histogram is None:
            histogram = None
        elif len(column_profile.histogram) > 10:
            histogram = None
        else:
            histogram = [
                {"value": item.value, "count": item.count, "ratio": round(item.ratio, 3)}
                for item in column_profile.histogram
            ]
        result_dict = {
            "completeness": column_profile.completeness,
            "approximateNumDistinctValues": column_profile.approximateNumDistinctValues,
            "dataType": column_profile.dataType,
            "typeCounts": column_profile.typeCounts,
            "isDataTypeInferred": column_profile.isDataTypeInferred,
            "histogram": histogram,
        }
        return result_dict
