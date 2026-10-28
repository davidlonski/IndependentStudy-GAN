from utils import send_email_notification, send_dcgan_completion_email
import os
from dotenv import load_dotenv

load_dotenv()

def test_basic_email():
    """Test basic email functionality"""
    print("Testing basic email notification...")
    
    recipient_email = os.getenv('EMAIL_RECIPIENT')  # Replace with your email
    
    success = send_email_notification(
        recipient_email=recipient_email,
        subject="Test Email - DCGAN Project",
        message="This is a test email to verify your email notification setup is working correctly!"
    )
    
    if success:
        print("✅ Basic email test passed!")
    else:
        print("❌ Basic email test failed!")

def test_dcgan_completion_email():
    """Test DCGAN completion email with sample data"""
    print("\nTesting DCGAN completion email...")
    
    recipient_email = os.getenv('EMAIL_RECIPIENT')  # Replace with your email
    
    # Sample training configuration
    sample_config = {
        'manual_seed': 42,
        'batch_size': 64,
        'image_size': 256,
        'num_epochs': 100,
        'lr': 0.00005
    }
    
    success = send_dcgan_completion_email(
        recipient_email=recipient_email,
        training_config=sample_config,
        training_time=3600,  # 1 hour in seconds
        output_dir="content/TRAIN"
    )
    
    if success:
        print("✅ DCGAN completion email test passed!")
    else:
        print("❌ DCGAN completion email test failed!")

def check_environment_variables():
    """Check if email environment variables are set"""
    print("Checking environment variables...")
    
    email_sender = os.getenv('EMAIL_SENDER')
    email_password = os.getenv('EMAIL_PASSWORD')
    
    if email_sender:
        print(f"✅ EMAIL_SENDER found: {email_sender}")
    else:
        print("❌ EMAIL_SENDER not found")
    
    if email_password:
        print(f"✅ EMAIL_PASSWORD found: {'*' * len(email_password)}")
    else:
        print("❌ EMAIL_PASSWORD not found")
    
    return email_sender and email_password

if __name__ == '__main__':
    print("=== Email Notification Test Script ===\n")
    
    # Check environment variables first
    env_ok = check_environment_variables()
    
    if not env_ok:
        print("\n⚠️  Environment variables not set. Please set EMAIL_SENDER and EMAIL_PASSWORD")
        print("You can set them in your terminal or create a .env file")
        exit(1)
    
    # Run tests
    test_basic_email()
    test_dcgan_completion_email()
    
    print("\n=== Test Complete ===")
    print("Check your email inbox for the test messages!") 