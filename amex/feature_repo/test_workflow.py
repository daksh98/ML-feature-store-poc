import subprocess
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from feast import FeatureStore
from feast.data_source import PushMode
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    SavedDatasetPostgreSQLStorage,
)
from feast.saved_dataset import ValidationReference
from feast.dqm.profilers.ge_profiler import ge_profiler
from great_expectations.dataset import PandasDataset
from great_expectations.core.expectation_suite import ExpectationSuite
from feast.dqm.errors import ValidationFailed

import psycopg2
from sqlalchemy import create_engine

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

def run_demo():
    store = FeatureStore(repo_path=".")

    # print("=====feast apply========")
    # subprocess.run(["feast", "apply"])

    # #####################
    # # get train data
    # #####################

    print("=====fetch train population + on demand ========")
    # feature_service 4 is for testing null conversions
    features_to_fetch = store.get_feature_service("customer_activity_v4")
    print(f"feature service owner ----> {features_to_fetch.owner} \ndescription ----> {features_to_fetch.description} \n")
    print()
    training_job = store.get_historical_features(
        entity_df=f"""
        select
            customer_id,
            s_2 as event_timestamp,
            target
        from
            amex_features
        where
            EXTRACT('Year' from s_2) = 2017""",
        features=features_to_fetch,
    )

    # dynamically generate a feature using on demand feature features
    # or just use the returned df to convert all nulls to 'N' when retriving data

    training_df = training_job.to_df()
    #training_df['s_6'].fillna('N' , inplace=True)
    print(training_df[training_df["s_6"].isna()])
    # #
    # # #####################
    # # # create reference
    # # #####################
    # print("=====save train population as reference========")
    #
    # # reference_dataset = store.create_saved_dataset(
    # #     from_=training_job,
    # #     name="reference_dataset",
    # #     storage=SavedDatasetPostgreSQLStorage("reference_dataset")
    # # )
    #
    # # #####################
    # # # model train
    # # #####################
    #
    # print("=====train model========")
    #
    # training_df.sort_values(by='event_timestamp', inplace = True)
    #
    # y = training_df["target"]
    # X = training_df.drop(['customer_id','target','event_timestamp'], axis=1)
    #
    # X = pd.get_dummies(X, columns=['d_63'], drop_first=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
    #
    # # Train a simple logistic  model
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    #
    # # Make a test prediction
    # y_pred = model.predict(X_test)
    # acc_score = model.score(X_test, y_test)
    #
    # print("")
    # print(f'acc score is {acc_score} f1_score is {f1_score(y_test, y_pred)} \n ')
    #
    # # import joblib
    # # joblib.dump(model, "model.bin")
    #
    # #####################
    # # online inference
    # #####################
    # print("=====online inference========")
    # #!feast materialize 2017-12-31T00:00:00 2018-01-01T00:00:00
    # store.materialize(start_date=datetime(2017, 11, 30, 00, 00, 00), end_date=datetime(2018, 1, 1, 00, 00, 00))
    # # funny business when retirving features with categories , as you need to create data set with all category dummies
    #     # a single customer may not have all the categories present for the feature value....
    #
    # # XZ 11861470386302782916
    # # CR 17277332996412040647
    # # CO 9466957447818569956
    # # CL 10811158664982877965
    # # XM 1011421575927340817
    # # XL 14554728156418915955
    # entity_rows = [
    #     {
    #         "customer_id": "11861470386302782916"
    #     },
    #     {
    #         "customer_id": "17277332996412040647"
    #     },
    #     {
    #         "customer_id": "9466957447818569956"
    #     },
    #     {
    #         "customer_id": "10811158664982877965"
    #     },
    #     {
    #         "customer_id": "1011421575927340817"
    #     },
    #     {
    #         "customer_id": "14554728156418915955"
    #     }
    #
    # ]
    # returned_features = store.get_online_features(
    #     features=features_to_fetch,
    #     entity_rows=entity_rows,
    # ).to_df()
    # print("here 112")
    # #new features dont have dummies
    # X_mat = pd.get_dummies(returned_features, columns=['d_63'], drop_first=True)
    # print(f"materialize data head \n {X.head()}")
    # X_mat.drop(['customer_id'], axis=1 , inplace=True)
    # X_mat = X_mat[X_train.columns]
    # print(f"Prediction from online customer is {model.predict(X_mat)}")
    #
    # ##########################################
    # # Data Validation on retiravl of CURRENT offline data
    # ###########################################
    #
    # print("=====Data Validation on retiravl of CURRENT offline data========")
    #
    # #create dataset profiler
    # reference_ds = store.get_saved_dataset('reference_dataset')
    # reference_ds.get_profile(profiler=stats_profiler)
    #
    # print(reference_ds.get_profile(profiler=stats_profiler))
    #
    # #Now we can create validation reference from dataset and profiler function:
    # #If retraival succeeds then data was validated if not data did not pass validation
    # try:
    #     training_df = training_job.to_df(
    #         validation_reference=store
    #         .get_saved_dataset("reference_dataset")
    #         .as_reference(profiler=stats_profiler , name='stats_profiler')
    #     )
    #     print("====success====")
    #     print(training_df.head())
    # except ValidationFailed as exc:
    #     print("====failure====")
    #     print(exc.validation_report)
    #
    #
    # # ##########################################
    # # # Data Validation on retiravl of NEW offline data
    # # ###########################################
    # print("=====Data Validation on retiravl of NEW offline data========")
    #
    # training_job_2 = store.get_historical_features(
    #     entity_df=f"""
    #     select
    #         customer_id,
    #         s_2 as event_timestamp,
    #         target
    #     from
    #         amex_features
    #     where
    #         EXTRACT('Year' from s_2) = 2018""",
    #     features=features_to_fetch,
    # )
    # try:
    #     training_df = training_job_2.to_df(
    #         validation_reference=store
    #         .get_saved_dataset("reference_dataset")
    #         .as_reference(profiler=stats_profiler , name='stats_profiler')
    #     )
    #     print("====success====")
    #     print(training_df.head())
    # except ValidationFailed as exc:
    #     print("====failure====")
    #     print(exc.validation_report)
    #

    ##########################################
    # Data Validation on retiravl of online data
    ###########################################
    # this can only be done thorugh a feature server and allows for traffic logging
    # see https://github.com/feast-dev/feast-gcp-fraud-tutorial/blob/main/notebooks/Validating_Online_Features_While_Detecting_Fraud.ipynb

    # Profiler function along with the reference dataset must be stored in the Feast registry before calling validation API:
    # ref = ValidationReference(
    #     name='user_features_training_ref',
    #     dataset_name="reference_dataset",
    #     profiler=stats_profiler,
    # )
    # store.apply(ref)

    #this command is online
    #! feast validate --feature-service customer_activity_v1 --reference user_features_training_ref 2018-03-31T00:00:00 2018-04-01T00:00:00

    ##########################################
    # Stimulate stream injestion
    ###########################################
    # print(
    #     "\n--- Online features retrieved (using feature service v3, which uses a feature view with a push source---"
    # )
    # fetch_online_features(store, source="push")
    #
    # print("\n--- Simulate a stream event ingestion of the hourly stats df ---")
    # event_df = pd.DataFrame.from_dict(
    #     {
    #         "driver_id": [1001],
    #         "event_timestamp": [
    #             datetime.now(),
    #         ],
    #         "created": [
    #             datetime.now(),
    #         ],
    #         "conv_rate": [1.0],
    #         "acc_rate": [1.0],
    #         "avg_daily_trips": [1000],
    #     }
    # )
    # print(event_df)
    # store.push("driver_stats_push_source", event_df, to=PushMode.ONLINE_AND_OFFLINE)
    #
    # print("\n--- Online features again with updated values from a stream push---")
    # fetch_online_features(store, source="push")
    #
    # print("\n--- Run feast teardown ---")
    # subprocess.run(["feast", "teardown"])


# Feast uses Great Expectations as a validation engine and ExpectationSuite as a dataset's profile.
# Hence, we need to develop a function that will generate ExpectationSuite.
# This function will receive instance of PandasDataset (wrapper around pandas.DataFrame) so we can utilize both Pandas DataFrame API
# and some helper functions from PandasDataset during profiling.

DELTA = 0.1  # controlling allowed window in fraction of the value on scale [0, 1]
@ge_profiler
def stats_profiler(ds: PandasDataset) -> ExpectationSuite:
    # simple checks on data consistency
    min_r_2 = ds['r_2'].min()
    max_r_2 = ds['r_2'].max()
    ds.expect_column_values_to_be_between(
        "r_2",
        min_value=min_r_2,
        max_value=max_r_2,
        mostly=0.99  # allow some outliers
    )

    min_b_4 = ds['b_4'].min()
    max_b_4 = ds['b_4'].max()
    ds.expect_column_values_to_be_between(
        "b_4",
        min_value=min_b_4,
        max_value=max_b_4,
        mostly=0.99  # allow some outliers
    )

    # expectation of means based on observed values
    observed_mean = ds.d_127.mean()
    ds.expect_column_mean_to_be_between("d_127",
                                        min_value=observed_mean * (1 - DELTA),
                                        max_value=observed_mean * (1 + DELTA))

    observed_mean = ds.d_65.mean()
    ds.expect_column_mean_to_be_between("d_65",
                                        min_value=observed_mean * (1 - DELTA),
                                        max_value=observed_mean * (1 + DELTA))

    return ds.get_expectation_suite()
    # # expectation of quantiles
    # qs = [0.5, 0.75, 0.9, 0.95]
    # observed_quantiles = ds.avg_fare.quantile(qs)
    #
    # ds.expect_column_quantile_values_to_be_between(
    #     "avg_fare",
    #     quantile_ranges={
    #         "quantiles": qs,
    #         "value_ranges": [[None, max_value] for max_value in observed_quantiles]
    #     })


if __name__ == "__main__":
    # try:
    #     engine = create_engine("postgresql+psycopg2://daksh:0762@localhost:5432/feast_data")
    #
    # except:
    #     print("I am unable to connect to the database")
    #
    # # we use a context manager to scope the cursor session
    # df = pd.read_sql_query("SELECT * FROM amex_features limit 10;", engine)
    # print(df)

    run_demo()
