# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64 , String

# Define an entity for the driver. You can think of an entity as a primary key used to fetch features.
customer = Entity( # can also defin type of join key
            name="customer",
            join_keys=["customer_id"],
            description="driver id",
            tags = {
                      "owner": "dakshmukhra1@gmail.com",
                      "owner_team": "CGRM",
                      "model": "Identity Crime Model",
                      "model_run_id": "1623891892",
                      "production": "True"
                    }
        )
