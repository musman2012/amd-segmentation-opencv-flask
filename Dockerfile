# start with a base image
FROM ubuntu

MAINTAINER Usman <musman14@student.bradford.ac.uk>

# install dependencies
RUN apt-get update
RUN apt-get install -y nginx supervisor
RUN apt-get install -y python3 python3-dev python3-pip python3-virtualenv
RUN apt-get install -y python3-opencv
RUN apt-get install -y python3-matplotlib
RUN apt-get install -y python3-scipy
RUN apt-get install -y python3-skimage
RUN apt-get install -y git
RUN pip3 install uwsgi
RUN pip3 install Flask
RUN pip3 install -U Werkzeug
RUN pip3 install Pillow
RUN pip3 install -U scikit-image

# update working directories
ADD ./app /app
ADD ./config /config

# setup config
RUN echo "\ndaemon off;" >> /etc/nginx/nginx.conf
RUN rm /etc/nginx/sites-enabled/default

RUN ln -s /config/nginx.conf /etc/nginx/sites-enabled/
RUN ln -s /config/supervisor.conf /etc/supervisor/conf.d/

EXPOSE 80
CMD ["python3", "app/app.py"]
