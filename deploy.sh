#!/bin/bash
set -e

echo "Starting static website deployment..."

# Ensure ownership is correct (optional, useful on EC2)
chown -R ec2-user:ec2-user .

# Optionally reload nginx if needed
# sudo systemctl reload nginx

echo "Static site deployed successfully."
