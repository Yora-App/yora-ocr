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

    # Ensure that the queue exists
    channel.queue_declare(queue="ocr_jobs_queue", durable=True)

    # Receiving a message
    # This function will print the message content
    def callback(ch, method, properties, body):
        print(f" [x] Received {body}")

    channel.basic_consume(
        queue="ocr_jobs_queue", on_message_callback=callback, auto_ack=True
    )

    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
