FROM python:3.11.9-slim-bookworm

ENV RUNNING_IN_DOCKER true
ENV DEBIAN_FRONTEND noninteractive

RUN apt clean \
	&& apt update \
	&& apt upgrade -y \
	&& apt-get install -y \
	ca-certificates \
	software-properties-common \
	sudo \
	bash \
	wget \
	curl \
	git \
	nano \
	htop \
	tree \
	cron \
	gcc \
	g++ \
	cmake \
	locales \
	sqlite3 \
	&& locale-gen en_US.UTF-8 \
	&& LC_ALL=en_US.UTF-8 \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG APP_USER_HOME=/root/home
WORKDIR $APP_USER_HOME

COPY ./src/backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN rm requirements.txt

RUN mkdir -p $APP_USER_HOME/logs