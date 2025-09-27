# Stage 1: Build
FROM rustlang/rust:nightly as builder

WORKDIR /usr/src/app
COPY . .

# Build in release mode
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/app/target/release/ai-service /app/ai-service
COPY models /app/models
COPY tokenizer /app/tokenizer

EXPOSE 5000

CMD ["./ai-service"]
