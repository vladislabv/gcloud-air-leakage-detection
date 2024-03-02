import logging
import sys
import os
import json

from pyflink.common import Types, Time
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.configuration import Configuration
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.datastream.connectors.kafka import FlinkKafkaProducer, FlinkKafkaConsumer
from pyflink.datastream.formats.json import JsonRowSerializationSchema, JsonRowDeserializationSchema


# # Make sure that the Kafka cluster is started and the topic 'test_json_topic' is
# # created before executing this job.
# def write_to_kafka(env):
#     type_info = Types.ROW([Types.INT(), Types.STRING()])
#     ds = env.from_collection(
#         [(1, 'hi'), (2, 'hello'), (3, 'hi'), (4, 'hello'), (5, 'hi'), (6, 'hello'), (6, 'hello')],
#         type_info=type_info)

#     serialization_schema = JsonRowSerializationSchema.Builder() \
#         .with_type_info(type_info) \
#         .build()
#     kafka_producer = FlinkKafkaProducer(
#         topic='test_json_topic',
#         serialization_schema=serialization_schema,
#         producer_config={'bootstrap.servers': 'localhost:9092', 'group.id': 'test_group'}
#     )

#     # note that the output type of ds must be RowTypeInfo
#     ds.add_sink(kafka_producer)
#     # env.execute()


# def read_from_kafka(env):
#     deserialization_schema = JsonRowDeserializationSchema.builder() \
#     .type_info(
#         type_info=Types.ROW(
#             #["index", "tmp_sec", "ft_position", "ft_type", "air_pressure", "tmp_min"],
#             [Types.LONG(), Types.STRING(), Types.STRING(), Types.STRING(), Types.DOUBLE(), Types.STRING()]
#         )
#     ).build()
#     kafka_consumer = FlinkKafkaConsumer(
#         topics='ft-sensor-test',
#         deserialization_schema=deserialization_schema,#SimpleStringSchema(),
#         properties={'bootstrap.servers': 'broker:29092', 'group.id': 'ft-sensor'}
#     )
#     kafka_consumer.set_start_from_earliest()

#     ds = env.add_source(kafka_consumer)

#     def update_json(data):
#         # parse the json
#         print(data)
#         json_data = json.loads(data)
#         json_data = {
#             'tmp_sec': json_data['tmp_sec'],
#             'ft_position': json_data['position'],
#             'ft_type': json_data['ft_type'],
#             'air_pressure': json_data['air_pressure']
#         }
#         return data[0], json_data

#     def filter_by_type(data):
#         # the json data could be accessed directly, there is no need to parse it again using
#         # json.loads
#         print(data)
#         return "FT1" == data[0]['ft_type']
    
#     serialization_schema = JsonRowSerializationSchema.builder().with_type_info(
#         type_info=Types.ROW(
#             #["tmp_sec", "ft_position", "ft_type", "air_pressure"],
#             [Types.STRING(), Types.STRING(), Types.STRING(), Types.DOUBLE()]
#         )
#     ).build()

#     ds.window_all(TumblingProcessingTimeWindows.of(Time.seconds(10))).apply(update_json, output_type=Types.ROW(
#             #["tmp_sec", "ft_position", "ft_type", "air_pressure"],
#             [Types.STRING(), Types.STRING(), Types.STRING(), Types.DOUBLE()]
#         )).print()
    
#     # env.execute()


def read_from_kafka_tbl(env, tbl_env):

    src_ddl = """
        CREATE TABLE ft_sensor (
            row_index INT,
            tmp_sec VARCHAR,
            ft_position BIGINT,
            ft_type VARCHAR,
            air_pressure DOUBLE,
            tmp_min VARCHAR,
            proctime AS PROCTIME()
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'ft-sensor',
            'properties.bootstrap.servers' = 'broker:29092',
            'properties.group.id' = 'ft-sensor',
            'format' = 'json'
        )
    """

    tbl_env.execute_sql(src_ddl)

    # create and initiate loading of source Table
    tbl = tbl_env.from_path('ft_sensor')

    print('\nSource Schema')
    tbl.print_schema()

    sql = """
        SELECT
          ft_type,
          TUMBLE_END(proctime, INTERVAL '60' SECONDS) AS window_end,
          SUM(air_pressure) AS window_air_pressure
        FROM ft_sensor
        GROUP BY
          TUMBLE(proctime, INTERVAL '60' SECONDS),
          ft_type
    """
    agg_tbl = tbl_env.sql_query(sql)

    print('\nProcess Sink Schema')
    agg_tbl.print_schema()

    sink_ddl = """
        CREATE TABLE ft_sensor_agg (
            ft_type VARCHAR,
            window_end TIMESTAMP(3),
            window_air_pressure DOUBLE
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'ft-sensor',
            'properties.bootstrap.servers' = 'broker:29092',
            'format' = 'json'
        )
    """
    tbl_env.execute_sql(sink_ddl)

    # write time windowed aggregations to sink table
    agg_tbl.execute_insert('ft_sensor_agg').wait().print()

    # tbl_env.execute()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars(f"file://{os.path.join(os.path.abspath(os.path.dirname(__file__)), 'flink-sql-connector-kafka-3.1.0-1.18.jar')}")

    configuration = Configuration()
    configuration.set_string(
        'pipeline.jars', f"file://{os.path.join(os.path.abspath(os.path.dirname(__file__)), 'flink-sql-connector-kafka-3.1.0-1.18.jar')}"
    )
    environment_settings = EnvironmentSettings \
        .new_instance() \
        .in_streaming_mode() \
        .with_configuration(configuration) \
        .build()
    
    # create table environment
    tbl_env = StreamTableEnvironment.create(
        stream_execution_environment=env,
        environment_settings=environment_settings
    )

    

    #print("start writing data to kafka")
    #write_to_kafka(env)

    print("start reading data from kafka")
    read_from_kafka_tbl(env, tbl_env)