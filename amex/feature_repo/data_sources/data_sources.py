# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64 , String

   # timestamp_field: Optional[str] = "",
   #      created_timestamp_column: Optional[str] = "",
   #      field_mapping: Optional[Dict[str, str]] = None,
   #      description: Optional[str] = "",
   #      tags: Optional[Dict[str, str]] = None,
   #      owner: Optional[str] = "",


# Define data sources for feature views
customer_data_source = PostgreSQLSource(
    name="customer_data_source",
    query="SELECT * FROM amex_features",
    timestamp_field="s_2",
    tags = {
              "owner": "dakshmukhra1@gmail.com",
              "owner_team": "CGRM",
              "production": "True"
            }
)
