# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64 , String

from feature_views.feature_views import *

customer_activity_v1 = FeatureService(
    name="customer_activity_v1",
    features=[
        customer_features,  # Can also Sub-selects a feature from a feature view
    ],
    tags = {
              "owner": "dakshmukhra1@gmail.com",
              "owner_team": "CGRM",
              "model": "Identity Crime Model",
              "model_run_id": "1623891892",
              "environment" : "staging"
            }
)

customer_activity_v2 = FeatureService(
    name="customer_activity_v2",
    features=[
        customer_features,  # Can also Sub-selects a feature from a feature view
        customer_features_transformed_B10,  # Selects all features from the feature view
    ],
    tags = {
              "owner": "dakshmukhra1@gmail.com",
              "owner_team": "CGRM",
              "model": "Incorrect Reporting Model",
              "model_run_id": "12639129123",
              "environment" : "staging"
            }
)


customer_activity_v4 = FeatureService(
    name="customer_activity_v4",
    features=[
        customer_features,
        customer_features_transformed_s_6,  # Can also Sub-selects a feature from a feature view
    ],
    # owner = "Daksh Mukhra",
    # description = "feature service to test feature trasnformation during retrieval",
    tags = {
              "owner": "dakshmukhra1@gmail.com",
              "owner_team": "CGRM",
              "model": "Identity Crime Model",
              "model_run_id": "1623891892",
              "environment" : "staging"
            }
)

driver_activity_v3 = FeatureService(
    name="customer_activity_v3",
    features=[customer_features_fresh, customer_features_transformed_B10_fresh],
    tags = {
              "owner": "dakshmukhra1@gmail.com",
              "owner_team": "CGRM",
              "model": "Incorrect Reporting Model",
              "model_run_id": "1623891892",
              "environment" : "production"
            }
)
