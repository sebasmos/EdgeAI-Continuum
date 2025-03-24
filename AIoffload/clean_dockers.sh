yes YES | docker rm -vf "$(docker ps -aq)"
yes YES | docker rmi -f $(docker images -aq)
yes YES | docker system prune -a
yes YES | docker system prune --all --force
yes YES | docker system prune --all --force --volumes