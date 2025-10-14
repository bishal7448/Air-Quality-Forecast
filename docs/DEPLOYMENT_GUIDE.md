# Air Quality Forecasting System - Deployment Architecture Guide

## Overview

This guide provides comprehensive instructions for deploying the Air Quality Forecasting System in various environments, from development to production scale. It covers containerization, cloud deployment, scaling strategies, and monitoring setup.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Environment Setup](#environment-setup)
3. [Docker Containerization](#docker-containerization)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Architecture](#production-architecture)
6. [Scaling Strategies](#scaling-strategies)
7. [Monitoring & Logging](#monitoring--logging)
8. [Security Considerations](#security-considerations)
9. [Maintenance & Updates](#maintenance--updates)
10. [Troubleshooting](#troubleshooting)

## Deployment Options

### 1. Local Development Deployment

**Use Case**: Development, testing, and demonstrations
**Requirements**: Single machine with Python 3.8+

```bash
# Quick setup for local development
cd air_quality_forecast
pip install -r requirements.txt
python setup.py  # Automated setup
python start_dashboard.py  # Launch web interface
```

### 2. Single Server Production

**Use Case**: Small-scale production, departmental use
**Requirements**: Linux server, 8GB RAM, 4 CPU cores

```bash
# Production server setup
sudo apt update && sudo apt upgrade -y
sudo apt install python3.8 python3-pip nginx supervisor -y
```

### 3. Container-based Deployment

**Use Case**: Scalable production, cloud deployment
**Requirements**: Docker, Kubernetes (optional)

```bash
# Docker deployment
docker build -t air-quality-forecast .
docker run -p 8501:8501 air-quality-forecast
```

### 4. Cloud-native Deployment

**Use Case**: High-availability, auto-scaling production
**Requirements**: AWS/GCP/Azure account

## Environment Setup

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 18.04+, CentOS 7+, Windows 10, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **CPU**: 4 cores (8 cores recommended)
- **Storage**: 50GB available space
- **Network**: Reliable internet connection for data updates

#### Production Requirements
- **OS**: Ubuntu 20.04 LTS (recommended)
- **Python**: 3.9+
- **RAM**: 32GB+ for large-scale deployment
- **CPU**: 16+ cores for concurrent training
- **Storage**: 500GB+ SSD storage
- **Network**: High-bandwidth connection, load balancer

### Python Environment

```bash
# Create virtual environment
python3 -m venv air_quality_env
source air_quality_env/bin/activate  # Linux/Mac
# air_quality_env\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost; print('All packages installed successfully')"
```

### Environment Variables

Create `.env` file for configuration:

```bash
# .env file
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database configuration (if using external DB)
DATABASE_URL=postgresql://user:password@localhost:5432/air_quality

# API keys (if using external services)
WEATHER_API_KEY=your_weather_api_key
SATELLITE_API_KEY=your_satellite_api_key

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=yourdomain.com,localhost

# Monitoring
MONITORING_ENABLED=true
SENTRY_DSN=your_sentry_dsn
```

## Docker Containerization

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Create necessary directories
RUN mkdir -p logs models results data \
    && chmod +x start_dashboard.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Default command
CMD ["python", "start_dashboard.py"]
```

### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  air-quality-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: air_quality
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
```

### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  air-quality-app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://app_user:${DB_PASSWORD}@postgres:5432/air_quality
    depends_on:
      - postgres
      - redis
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - air-quality-app
    restart: always
    networks:
      - app-network

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: air_quality
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always
    networks:
      - app-network

  redis:
    image: redis:alpine
    restart: always
    networks:
      - app-network

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: always
    networks:
      - app-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: always
    networks:
      - app-network

volumes:
  postgres_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS (Elastic Container Service)

```yaml
# aws-ecs-task-definition.json
{
  "family": "air-quality-forecast",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "air-quality-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/air-quality-forecast:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:air-quality-db"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/air-quality-forecast",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8501/_stcore/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### CloudFormation Template

```yaml
# cloudformation-template.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Air Quality Forecasting System Infrastructure'

Parameters:
  EnvironmentName:
    Description: Environment name (dev, staging, prod)
    Type: String
    Default: prod

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-air-quality-vpc

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.2.0/24
      MapPublicIpOnLaunch: true

  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub ${EnvironmentName}-air-quality-cluster

  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub ${EnvironmentName}-air-quality-lb
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup

  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: !Sub ${EnvironmentName}-air-quality-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref AppSecurityGroup
          Subnets:
            - !Ref PublicSubnet1
            - !Ref PublicSubnet2
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: air-quality-app
          ContainerPort: 8501
          TargetGroupArn: !Ref TargetGroup
```

### GCP Deployment (Google Cloud Platform)

#### Using Cloud Run

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/air-quality-forecast
gcloud run deploy air-quality-forecast \
  --image gcr.io/PROJECT-ID/air-quality-forecast \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

#### Cloud Deployment Manager Template

```yaml
# gcp-deployment.yaml
resources:
- name: air-quality-cluster
  type: container.v1.cluster
  properties:
    zone: us-central1-a
    cluster:
      name: air-quality-cluster
      initialNodeCount: 3
      nodeConfig:
        machineType: n1-standard-4
        diskSizeGb: 100
        oauthScopes:
        - https://www.googleapis.com/auth/compute
        - https://www.googleapis.com/auth/devstorage.read_only
        - https://www.googleapis.com/auth/logging.write
        - https://www.googleapis.com/auth/monitoring

- name: air-quality-service
  type: apps/v1/Deployment
  properties:
    namespace: default
    metadata:
      name: air-quality-service
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: air-quality-forecast
      template:
        metadata:
          labels:
            app: air-quality-forecast
        spec:
          containers:
          - name: air-quality-app
            image: gcr.io/PROJECT-ID/air-quality-forecast:latest
            ports:
            - containerPort: 8501
            resources:
              requests:
                memory: "2Gi"
                cpu: "1000m"
              limits:
                memory: "4Gi"
                cpu: "2000m"
```

### Azure Deployment

#### Using Azure Container Instances

```bash
# Deploy to Azure Container Instances
az group create --name air-quality-rg --location eastus

az container create \
  --resource-group air-quality-rg \
  --name air-quality-forecast \
  --image your-registry.azurecr.io/air-quality-forecast:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501 \
  --dns-name-label air-quality-forecast \
  --environment-variables ENVIRONMENT=production
```

## Production Architecture

### High-Availability Architecture

```
                                    ┌─────────────────┐
                                    │   Load Balancer │
                                    │    (AWS ALB)    │
                                    └─────────┬───────┘
                                              │
                         ┌────────────────────┼────────────────────┐
                         │                    │                    │
                ┌────────▼──────┐    ┌────────▼──────┐    ┌────────▼──────┐
                │ App Instance 1│    │ App Instance 2│    │ App Instance 3│
                │   (ECS/K8s)   │    │   (ECS/K8s)   │    │   (ECS/K8s)   │
                └───────────────┘    └───────────────┘    └───────────────┘
                         │                    │                    │
                         └────────────────────┼────────────────────┘
                                              │
    ┌─────────────────┐                      │                      ┌─────────────────┐
    │     Redis       │◄─────────────────────┼─────────────────────►│   PostgreSQL    │
    │   (Caching)     │                      │                      │   (Database)    │
    └─────────────────┘                      │                      └─────────────────┘
                                              │
    ┌─────────────────┐                      │                      ┌─────────────────┐
    │   Prometheus    │◄─────────────────────┼─────────────────────►│    Grafana      │
    │  (Monitoring)   │                      │                      │  (Dashboard)    │
    └─────────────────┘                      │                      └─────────────────┘
                                              │
    ┌─────────────────┐                      │                      ┌─────────────────┐
    │  File Storage   │◄─────────────────────┼─────────────────────►│   Log Storage   │
    │   (S3/GCS)      │                      │                      │ (CloudWatch)    │
    └─────────────────┘                      │                      └─────────────────┘
                                              │
                                    ┌────────▼──────┐
                                    │   API Gateway │
                                    │   (Optional)  │
                                    └───────────────┘
```

### Microservices Architecture

For large-scale deployments, consider breaking the system into microservices:

#### Service Breakdown

1. **Data Ingestion Service**
   - Handles data loading and preprocessing
   - Exposes REST API for data operations
   - Manages data quality checks

2. **Feature Engineering Service**
   - Creates and manages features
   - Caching for computed features
   - Feature store integration

3. **Model Training Service**
   - Handles model training workflows
   - Model versioning and storage
   - Training job orchestration

4. **Prediction Service**
   - Real-time prediction API
   - Model serving and inference
   - Response caching

5. **Dashboard Service**
   - Web interface
   - Visualization components
   - User management

#### Service Communication

```yaml
# kubernetes-services.yaml
apiVersion: v1
kind: Service
metadata:
  name: data-ingestion-service
spec:
  selector:
    app: data-ingestion
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: prediction-service
spec:
  selector:
    app: prediction-service
  ports:
    - port: 8001
      targetPort: 8001
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
spec:
  selector:
    app: dashboard
  ports:
    - port: 8501
      targetPort: 8501
  type: LoadBalancer
```

## Scaling Strategies

### Horizontal Scaling

#### Kubernetes Auto-scaling

```yaml
# hpa.yaml (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: air-quality-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: air-quality-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### AWS Auto Scaling

```yaml
# aws-autoscaling-policy.json
{
  "AutoScalingGroupName": "air-quality-asg",
  "PolicyName": "air-quality-scale-policy",
  "AdjustmentType": "ChangeInCapacity",
  "ScalingAdjustment": 2,
  "Cooldown": 300,
  "PolicyType": "SimpleScaling"
}
```

### Vertical Scaling

#### Resource Optimization

```yaml
# kubernetes-deployment-optimized.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: air-quality-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: air-quality-forecast
  template:
    metadata:
      labels:
        app: air-quality-forecast
    spec:
      containers:
      - name: air-quality-app
        image: air-quality-forecast:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        env:
        - name: WORKERS
          value: "4"
        - name: MEMORY_LIMIT
          value: "6Gi"
```

### Database Scaling

#### PostgreSQL High Availability

```yaml
# postgres-ha.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      
  bootstrap:
    initdb:
      database: air_quality
      owner: app_user
      secret:
        name: postgres-credentials
        
  storage:
    size: 500Gi
    storageClass: fast-ssd
    
  monitoring:
    enabled: true
```

## Monitoring & Logging

### Application Monitoring

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'air-quality-app'
    static_configs:
      - targets: ['air-quality-app:8501']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Custom Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('air_quality_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('air_quality_request_duration_seconds', 'Request duration')
MODEL_ACCURACY = Gauge('air_quality_model_accuracy', 'Current model accuracy', ['model_type'])
PREDICTION_COUNT = Counter('air_quality_predictions_total', 'Total predictions made')

class Metrics:
    def __init__(self):
        start_http_server(8000)  # Metrics endpoint
    
    def record_request(self, method, endpoint, duration):
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_DURATION.observe(duration)
    
    def update_model_accuracy(self, model_type, accuracy):
        MODEL_ACCURACY.labels(model_type=model_type).set(accuracy)
    
    def record_prediction(self):
        PREDICTION_COUNT.inc()
```

### Logging Configuration

#### Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
```

#### Log Aggregation (ELK Stack)

```yaml
# elasticsearch.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## Security Considerations

### Network Security

#### Network Policies (Kubernetes)

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: air-quality-network-policy
spec:
  podSelector:
    matchLabels:
      app: air-quality-forecast
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx
    ports:
    - protocol: TCP
      port: 8501
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Application Security

#### Security Headers

```python
# security/middleware.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

def add_security_middleware(app: FastAPI):
    # HTTPS redirect
    app.add_middleware(HTTPSRedirectMiddleware)
    
    # Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### Secrets Management

#### AWS Secrets Manager Integration

```python
# security/secrets.py
import boto3
import json
from botocore.exceptions import ClientError

class SecretsManager:
    def __init__(self, region_name="us-west-2"):
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    def get_secret(self, secret_name):
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except ClientError as e:
            raise e
    
    def get_database_credentials(self):
        return self.get_secret("air-quality/database")
    
    def get_api_keys(self):
        return self.get_secret("air-quality/api-keys")
```

## Maintenance & Updates

### Continuous Integration/Continuous Deployment

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Air Quality Forecast

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Run security scan
      run: |
        pip install bandit
        bandit -r src/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build, tag, and push image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: air-quality-forecast
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster air-quality-cluster \
          --service air-quality-service --force-new-deployment
```

### Database Migrations

#### Migration Scripts

```python
# migrations/migrate.py
import psycopg2
import os
from pathlib import Path

class DatabaseMigrator:
    def __init__(self, database_url):
        self.database_url = database_url
        self.migrations_path = Path("migrations/sql")
    
    def run_migrations(self):
        conn = psycopg2.connect(self.database_url)
        cur = conn.cursor()
        
        # Create migrations table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Get applied migrations
        cur.execute("SELECT filename FROM migrations")
        applied = set(row[0] for row in cur.fetchall())
        
        # Apply new migrations
        migration_files = sorted(self.migrations_path.glob("*.sql"))
        for migration_file in migration_files:
            if migration_file.name not in applied:
                print(f"Applying migration: {migration_file.name}")
                
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()
                
                cur.execute(migration_sql)
                cur.execute(
                    "INSERT INTO migrations (filename) VALUES (%s)",
                    (migration_file.name,)
                )
                conn.commit()
        
        cur.close()
        conn.close()
        print("All migrations completed successfully")

if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    migrator = DatabaseMigrator(database_url)
    migrator.run_migrations()
```

### Backup and Recovery

#### Database Backup Script

```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backups"
DB_NAME="air_quality"
DB_USER="app_user"
S3_BUCKET="air-quality-backups"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename with timestamp
BACKUP_FILE="$BACKUP_DIR/air_quality_$(date +%Y%m%d_%H%M%S).sql"

# Create database backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3
aws s3 cp "$BACKUP_FILE.gz" "s3://$S3_BUCKET/database/"

# Clean up old local backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

# Clean up old S3 backups
aws s3 ls "s3://$S3_BUCKET/database/" | while read -r line; do
    FILE_DATE=$(echo $line | awk '{print $1}')
    FILE_NAME=$(echo $line | awk '{print $4}')
    
    if [[ $(date -d "$FILE_DATE" +%s) -lt $(date -d "$RETENTION_DAYS days ago" +%s) ]]; then
        aws s3 rm "s3://$S3_BUCKET/database/$FILE_NAME"
    fi
done

echo "Backup completed: $BACKUP_FILE.gz"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

**Problem**: Application consuming too much memory
**Solution**:
```bash
# Check memory usage
kubectl top pods
docker stats

# Optimize memory settings
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export PANDAS_MAX_ROWS=10000

# Update resource limits
kubectl patch deployment air-quality-deployment -p '{"spec":{"template":{"spec":{"containers":[{"name":"air-quality-app","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

#### 2. Slow Predictions

**Problem**: Model predictions taking too long
**Solutions**:
```python
# Enable model caching
@lru_cache(maxsize=1000)
def predict_cached(features_hash):
    return model.predict(features)

# Use batch prediction
predictions = model.predict(batch_features)

# Optimize feature engineering
# Use vectorized operations instead of loops
```

#### 3. Database Connection Issues

**Problem**: Database connection failures
**Solution**:
```python
# Implement connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Add retry logic
import time
from functools import wraps

def retry_db_connection(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))
            return wrapper
    return decorator
```

### Health Checks

#### Comprehensive Health Check

```python
# health.py
from fastapi import FastAPI, HTTPException
import psycopg2
import redis
import requests
import time

app = FastAPI()

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    overall_healthy = True
    
    # Database health
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        overall_healthy = False
    
    # Redis health
    try:
        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        overall_healthy = False
    
    # Model loading health
    try:
        # Check if models are loaded
        if hasattr(app.state, 'models') and app.state.models:
            health_status["checks"]["models"] = "healthy"
        else:
            health_status["checks"]["models"] = "unhealthy: models not loaded"
            overall_healthy = False
    except Exception as e:
        health_status["checks"]["models"] = f"unhealthy: {str(e)}"
        overall_healthy = False
    
    if not overall_healthy:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@app.get("/ready")
async def readiness_check():
    # Simplified readiness check
    return {"status": "ready", "timestamp": time.time()}
```

This comprehensive deployment guide provides everything needed to deploy the Air Quality Forecasting System from development to large-scale production environments, with proper monitoring, security, and maintenance procedures.
