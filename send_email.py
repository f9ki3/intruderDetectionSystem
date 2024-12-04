import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email content setup
sender_email = "carurucankylejustin@gmail.com"
receiver_email = "rickandmorty0224@gmail.com"
subject = "Hello from Python"
body = "This is a test email sent using Python, and it includes an image attachment!"

# Create the email message
msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = subject

# Add email body
msg.attach(MIMEText(body, "plain"))

# Add the image attachment
attachment_path = "screenshots/wall1.jpeg"
try:
    with open(attachment_path, "rb") as attachment_file:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment_file.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={attachment_path.split('/')[-1]}"
        )
        msg.attach(part)
except FileNotFoundError:
    print(f"Error: File {attachment_path} not found. Skipping attachment.")

# Sending the email
try:
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, "fyxx aaoc rqqr tzkt")  # App Password
        server.send_message(msg)
        print("Email sent successfully with the image attachment!")
except Exception as e:
    print(f"Error sending email: {e}")
