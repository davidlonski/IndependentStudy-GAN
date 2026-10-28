import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

def send_email_notification(
    recipient_email,
    subject="DCGAN Training Complete",
    message="Your DCGAN training has finished successfully!",
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    sender_email=None,
    sender_password=None
):
    """
    Send an email notification when training is complete.
    
    Args:
        recipient_email (str): Email address to send notification to
        subject (str): Email subject line
        message (str): Email message body
        smtp_server (str): SMTP server (default: Gmail)
        smtp_port (int): SMTP port (default: 587 for TLS)
        sender_email (str): Sender email address (if None, will try to get from environment)
        sender_password (str): Sender password/app password (if None, will try to get from environment)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get credentials from environment variables if not provided
        if sender_email is None:
            sender_email = os.getenv('EMAIL_SENDER')
        if sender_password is None:
            sender_password = os.getenv('EMAIL_PASSWORD')
            
        if not sender_email or not sender_password:
            print("Warning: Email credentials not found. Please set EMAIL_SENDER and EMAIL_PASSWORD environment variables.")
            print("Or provide sender_email and sender_password parameters.")
            return False
            
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add timestamp to message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{message}\n\nTraining completed at: {timestamp}"
        
        msg.attach(MIMEText(full_message, 'plain'))
        
        # Create secure SSL/TLS context
        context = ssl.create_default_context()
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            
        print(f"Email notification sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send email notification: {str(e)}")
        return False

def send_dcgan_completion_email(
    recipient_email,
    training_config=None,
    training_time=None,
    output_dir=None
):
    """
    Send a specific email notification for DCGAN training completion.
    
    Args:
        recipient_email (str): Email address to send notification to
        training_config (dict): Training configuration parameters
        training_time (float): Total training time in seconds
        output_dir (str): Output directory where results were saved
    """
    subject = "🎉 DCGAN Training Complete!"
    
    # Build message
    message_parts = [
        "Your DCGAN training has finished successfully! 🎉",
        "",
        "Training Summary:"
    ]
    
    if training_config:
        message_parts.append("Configuration:")
        for key, value in training_config.items():
            message_parts.append(f"  • {key}: {value}")
        message_parts.append("")
    
    if training_time:
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"
        message_parts.append(f"Total Training Time: {time_str}")
        message_parts.append("")
    
    if output_dir:
        message_parts.append(f"Results saved to: {output_dir}")
        message_parts.append("")
    
    message_parts.extend([
        "You can now check your generated images and training results.",
        "",
        "Happy generating! 🚀"
    ])
    
    message = "\n".join(message_parts)
    
    return send_email_notification(recipient_email, subject, message)
