# Stage 1: Build the Flutter application
FROM cirrusci/flutter:stable AS build

# Set environment variables
ENV FLUTTER_HOME /flutter
ENV PATH $FLUTTER_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy pubspec.yaml and pubspec.lock to install dependencies
COPY pubspec.* ./

# Install Flutter dependencies
RUN flutter pub get

# Copy the entire project
COPY . .

# Build the Flutter web app for production
RUN flutter build web --release

# Stage 2: Serve the Flutter application
FROM nginx:alpine

# Copy built Flutter web app from previous stage
COPY --from=build /app/build/web /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Command to start NGINX
CMD ["nginx", "-g", "daemon off;"]
