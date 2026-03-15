from staging_pipeline import build_staging_layer
from analytics_pipeline import build_analytics_layer

if __name__ == "__main__":
    build_staging_layer()
    build_analytics_layer()
