# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64 , String

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
customer = Entity(name="customer",join_keys=["customer_id"])

customer_data_source = PostgreSQLSource(
    name="customer_data_source",
    query="SELECT * FROM amex_features",
    timestamp_field="s_2"
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
customer_stats_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="customer_features",
    entities=[customer],
    ttl=timedelta(weeks=10),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="d_63", dtype=String),
        Field(name="s_6", dtype=Float32),
        Field(name="r_16", dtype=Float32),
        Field(name="b_10", dtype=Float32),
        Field(name="r_1", dtype=Float32),
        Field(name="d_127", dtype=Float32),
        Field(name="b_36", dtype=Float32),
        Field(name="d_65", dtype=Float32),
        Field(name="r_2", dtype=Float32),
        Field(name="b_4", dtype=Float32),
        Field(name="d_92", dtype=Float32),
    ],

    online=True,
    source=customer_data_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"teams_tag": "customer_rating"},
)

# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
input_request = RequestSource(
    name="vals_to_add",
    schema=[
        Field(name="val_to_add", dtype=Float32),
        Field(name="val_to_add_2", dtype=Float32),
    ],
)

# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
@on_demand_feature_view(
    sources=[customer_stats_fv, input_request],
    schema=[
        Field(name="b_10_plus_val1", dtype=Float64),
        Field(name="b_10_plus_val2", dtype=Float64),
    ],
)
def transformed_B10(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["b_10_plus_val1"] = inputs["b_10"] + inputs["val_to_add"]
    df["b_10_plus_val2"] = inputs["b_10"] + inputs["val_to_add_2"]
    return df


# This groups features into a model version
# can select featrues from different feature views
customer_activity_v1 = FeatureService(
    name="customer_activity_v1",
    features=[
        customer_stats_fv,  # Can also Sub-selects a feature from a feature view
    ]
)

customer_activity_v2 = FeatureService(
    name="customer_activity_v2",
    features=[
        customer_stats_fv,  # Can also Sub-selects a feature from a feature view
        transformed_B10,  # Selects all features from the feature view
    ]
)
# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
input_request_2 = RequestSource(
    name="null_to_ns",
    schema=[
        Field(name="val_is_n", dtype=String),
    ],
)

# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
@on_demand_feature_view(
    sources=[customer_stats_fv, input_request_2],
    schema=[
        Field(name="s_6_to_n", dtype=Float64),
    ],
)
def transformed_s_6(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()

    df['s_6_to_n'] = inputs['s_6'].fillna('N')
    return df

customer_activity_v4 = FeatureService(
    name="customer_activity_v4",
    features=[
        customer_stats_fv,
        transformed_s_6,  # Can also Sub-selects a feature from a feature view
    ]
)


# Defines a way to push data (to be available offline, online or both) into Feast.
customer_stats_push_source = PushSource(
    name="customer_stats_push_source",
    batch_source=customer_data_source,
)

# Defines a slightly modified version of the feature view from above, where the source
# has been changed to the push source. This allows fresh features to be directly pushed
# to the online store for this feature view.
customer_stats_fresh_fv = FeatureView(
    name="customer_stats_fresh",
    entities=[customer],
    ttl=timedelta(weeks=10),
    schema=[
        Field(name="d_63", dtype=String),
        Field(name="s_6", dtype=Float32),
        Field(name="r_16", dtype=Float32),
        Field(name="b_10", dtype=Float32),
        Field(name="r_1", dtype=Float32),
        Field(name="d_127", dtype=Float32),
        Field(name="b_36", dtype=Float32),
        Field(name="d_65", dtype=Float32),
        Field(name="r_2", dtype=Float32),
        Field(name="b_4", dtype=Float32),
        Field(name="d_92", dtype=Float32),
    ],
    online=True,
    source=customer_stats_push_source,
    tags={"teams_tag": "customer_rating"},
)


# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
@on_demand_feature_view(
    sources=[customer_stats_fresh_fv, input_request],  # relies on fresh version of FV
    schema=[
        Field(name="b_10_plus_val1", dtype=Float64),
        Field(name="b_10_plus_val2", dtype=Float64),
    ],
)
def transformed_B10_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["b_10_plus_val1"] = inputs["b_10"] + inputs["val_to_add"]
    df["b_10_plus_val2"] = inputs["b_10"] + inputs["val_to_add_2"]
    return df


driver_activity_v3 = FeatureService(
    name="customer_activity_v3",
    features=[customer_stats_fresh_fv, transformed_B10_fresh]
)
