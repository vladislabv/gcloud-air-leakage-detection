import csv
import time
import json
from google.cloud import pubsub_v1
from google.auth import jwt

service_account_info = json.load(open("service-account-info.json"))
audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"

credentials = jwt.Credentials.from_service_account_info(
    service_account_info, audience=audience
)

topic_name = 'projects/{project_id}/topics/{topic}'.format(
    project_id="academic-oath-414215",
    topic='ft_sensor',
)

# now read messages as csv and publish them once in 15 seconds
with pubsub_v1.PublisherClient(credentials=credentials) as publisher:
    with open('druck_values.csv', mode='r') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            print(row)
            record = json.dumps(row).encode("utf-8")
            future = publisher.publish(topic_name, record)
            print("Message was published under id: {}".format(future.result()))
            time.sleep(15)



