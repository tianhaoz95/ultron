FROM gitpod/workspace-full

RUN apt update -y
RUN apt install -y python-opengl
RUN apt install -y xvfb
