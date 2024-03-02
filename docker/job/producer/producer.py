import argparse
import atexit
import json
import logging
import csv
import time
import sys


from confluent_kafka import Producer


logging.basicConfig(
  format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[
      logging.FileHandler("sensor_data_producer.log"),
      logging.StreamHandler(sys.stdout)
  ]
)

logger = logging.getLogger()


class ProducerCallback:
    def __init__(self, record, log_success=False):
        self.record = record
        self.log_success = log_success

    def __call__(self, err, msg):
        if err:
            logger.error('Error producing record {}'.format(self.record))
        elif self.log_success:
            logger.info('Produced {} to topic {} partition {} offset {}'.format(
                self.record,
                msg.topic(),
                msg.partition(),
                msg.offset()
            ))


def main(args):
    logger.info('Starting sensor data producer')
    conf = {
        'bootstrap.servers': args.bootstrap_server,
        'linger.ms': 200,
        'client.id': 'ft-sensor',
        'partitioner': 'murmur2_random'
    }

    producer = Producer(conf)

    atexit.register(lambda p: p.flush(), producer)
    
    with open(args.file, mode='r') as infile:
        i = 1
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            is_tenth = i % 10 == 0

            producer.produce(topic=args.topic,
                            value=json.dumps(row).encode("utf-8"),
                            on_delivery=ProducerCallback(row, log_success=is_tenth))

            if is_tenth:
                producer.poll(1)
                time.sleep(5)
                i = 0 # no need to let i grow unnecessarily large

            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-server', default='broker:29092')
    parser.add_argument('--topic', default='ft-sensor')
    parser.add_argument('--file', default='druck_values.csv')
    args = parser.parse_args()
    main(args)