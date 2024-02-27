import json
import base64
from google.cloud import pubsub_v1
from google.auth import jwt

service_account_info = json.load(open("service-account-info.json"))
audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"

credentials = jwt.Credentials.from_service_account_info(
    service_account_info, audience=audience
)

subscription_name = 'projects/{project_id}/subscriptions/{sub}'.format(
    project_id="academic-oath-414215",
    sub='ft_sensor-sub',
)

def callback(message):
    print("Ping! Message came in!")
    message.ack()
    try:
        data_js = json.loads(message.data.decode())
        print(f"Decoded data {data_js}")
    except Exception:
        print(message.data)

with pubsub_v1.SubscriberClient(credentials=credentials) as subscriber:
    future = subscriber.subscribe(subscription_name, callback)
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()