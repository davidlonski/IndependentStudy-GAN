# Email Notification Setup Guide

This guide explains how to set up email notifications for when your DCGAN training completes.

## Prerequisites

1. **Gmail Account** (recommended for simplicity)
2. **App Password** (not your regular password)

## Setup Steps

### 1. Enable 2-Factor Authentication
- Go to your Google Account settings
- Enable 2-Factor Authentication if not already enabled

### 2. Generate App Password
- Go to Google Account → Security → App passwords
- Select "Mail" and your device
- Copy the generated 16-character password

### 3. Set Environment Variables

#### Option A: Set in your system environment
```bash
# Windows (PowerShell)
$env:EMAIL_SENDER="your-email@gmail.com"
$env:EMAIL_PASSWORD="your-16-char-app-password"
$env:NOTIFICATION_EMAIL="your-email@gmail.com"

# Windows (Command Prompt)
set EMAIL_SENDER=your-email@gmail.com
set EMAIL_PASSWORD=your-16-char-app-password
set NOTIFICATION_EMAIL=your-email@gmail.com

# Linux/Mac
export EMAIL_SENDER="your-email@gmail.com"
export EMAIL_PASSWORD="your-16-char-app-password"
export NOTIFICATION_EMAIL="your-email@gmail.com"
```

#### Option B: Create a .env file (recommended)
Create a file named `.env` in your project root: 