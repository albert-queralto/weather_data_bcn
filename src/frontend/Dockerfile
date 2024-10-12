FROM node:lts-slim

ARG APP_USER_HOME=/root/home
WORKDIR $APP_USER_HOME

ENV PATH /$APP_USER_HOME/node_modules/.bin:$PATH

COPY package.json /$APP_USER_HOME/package.json
RUN npm install

WORKDIR /$APP_USER_HOME/frontend