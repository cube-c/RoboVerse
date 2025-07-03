#!/bin/bash

# Simple RoboVerse container management script

set -e

# Set environment variables
export NVIDIA_API_KEY=${NVIDIA_API_KEY:-}
export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)
export DOCKER_USER=${USER}

# Handle commands
case "${1:-start}" in
    start)
        echo "Starting RoboVerse container..."

        # Check NVIDIA API key
        if [ -z "$NVIDIA_API_KEY" ]; then
            echo "Warning: NVIDIA_API_KEY is not set."
            echo "export NVIDIA_API_KEY=your_key_here"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi

        # Clean up existing container
        docker stop metasim 2>/dev/null || true
        docker rm metasim 2>/dev/null || true

        # Build and run
        docker compose up -d --build

        # Check status
        echo "Container started successfully!"
        docker ps | grep metasim
        echo ""
        echo "Access: $0 exec"
        echo "Stop: $0 stop"
        ;;

    stop)
        echo "Stopping RoboVerse container..."
        docker compose down
        echo "Container stopped successfully!"
        ;;

    exec)
        echo "Accessing container..."
        docker exec -it metasim bash
        ;;

    logs)
        echo "Container logs:"
        docker logs metasim
        ;;

    restart)
        echo "Restarting container..."
        $0 stop
        sleep 2
        $0 start
        ;;

    *)
        echo "RoboVerse Container Management Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start    Start container (default)"
        echo "  stop     Stop container"
        echo "  restart  Restart container"
        echo "  exec     Access container"
        echo "  logs     Show container logs"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 exec"
        echo "  $0 stop"
        ;;
esac
