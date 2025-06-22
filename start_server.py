#!/usr/bin/env python3
"""
Server Starter for PayPal Fraud Detection API
========================================================

This script starts the FastAPI server with configurable options.
It provides a production-ready entry point for the fraud detection API.
"""

import uvicorn
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description='Start the PayPal Fraud Detection API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"🚀 Starting PayPal Fraud Detection API")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Alternative docs: http://{args.host}:{args.port}/redoc")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
