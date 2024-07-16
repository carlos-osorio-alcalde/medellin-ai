# Start docker
open -a Docker

# Pull qdrant image
docker pull qdrant/qdrant

# Run qdrant container
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant