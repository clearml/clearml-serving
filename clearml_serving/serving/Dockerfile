FROM python:3.11-bullseye


ENV LC_ALL=C.UTF-8

# install base package
RUN pip3 install --no-cache-dir clearml-serving

# get latest execution code from the git repository
# RUN cd $HOME && git clone https://github.com/allegroai/clearml-serving.git
COPY clearml_serving /root/clearml/clearml_serving

RUN pip3 install --no-cache-dir -r /root/clearml/clearml_serving/serving/requirements.txt

# default serving port
EXPOSE 8080

# environement variable to load Task from CLEARML_SERVING_TASK_ID, CLEARML_SERVING_PORT

WORKDIR /root/clearml/
ENTRYPOINT ["clearml_serving/serving/entrypoint.sh"]
