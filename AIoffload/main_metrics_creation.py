from model.metrics_utilisation import MetricsUtilisation
from dataclay import Client
from dataclay.exceptions import AlreadyExistError

# This is intended to be run within the docker-compose
client = Client(proxy_host="proxy", dataset="testdata")
client.start()

persistent_mt = MetricsUtilisation()
try:
    persistent_mt.make_persistent(alias="main_metrics")
except AlreadyExistError:
    print("Alias already exists, do nothing")