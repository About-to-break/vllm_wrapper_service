from logging_tools import logging_tools
from rabbitmq_tools.rabbitmq import RabbitProducer, RabbitConsumer
from minio_tools import minio_client
from internal.pipeline import pipeline
from config import load_config



def serve():
    config = load_config()
    logger = logging_tools.get_logger(
        "Scene detector",
        level=config.LOG_LEVEL,
        file=config.LOG_FILE
    )
    logger.info("Starting server")

    file_client = minio_client.MinioClient(
        endpoint=config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=config.MINIO_USE_SSL,
        bucket_name=config.MINIO_BUCKET,
        logger=logger
    )

    pipe = pipeline.MainPipeline(
        logger=logger,
        roi_device=config.ULTRA_DEVICE,
        minio_client=file_client,
        model=config.ULTRA_MODEL_PATH,
        roi_target_class=config.ULTRA_TARGET_CLASS,
    )

    consumer = RabbitConsumer(
        uri=config.RABBITMQ_URI,
        queue=config.RABBITMQ_QUEUE,
        logger=logger,
    )

    producer = RabbitProducer(
        uri=config.RABBITMQ_URI,
        key=config.RABBITMQ_ROUTING_KEY,
        exchange=config.RABBITMQ_EXCHANGE,
        logger=logger,
    )

    logger.info("Connecting to services...")

    consumer.connect()
    producer.connect()

    logger.info("Init server DONE")

    consumer.consume(pipe.run)

if __name__ == '__main__':
    serve()
