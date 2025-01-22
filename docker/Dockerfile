FROM python:3.12

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y plastimatch
RUN apt-get install -y git
RUN pip uninstall -y SimpleITK
RUN pip uninstall SimpleITK-SimpleElastix
RUN pip install SimpleITK-SimpleElastix