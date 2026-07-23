import pika
import os
import sys


def main():

    credentials = pika.PlainCredentials(username="admin", password="secure_password")

    # Create connection
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost", credentials=credentials)
    )
    channel = connection.channel()

    # Set QoS prefetch_count to one, so that rabbitmq only assigns a message to this worker
    # after it acked the last one
    channel.basic_qos(prefetch_count=1)

    # Ensure that the queue exists
    channel.queue_declare(queue="ocr_jobs_queue", durable=True)

    print(" [*] Waiting for messages. To exit press CTRL+C")
    for method, properties, body in channel.consume("ocr_jobs_queue"):
        if (method is not None and properties is not None and body is not None):
            # placeholder: put ocr here before ack
            print(f" [x] Received {body}")

            channel.basic_ack(method.delivery_tag)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
