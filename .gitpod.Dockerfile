FROM gitpod/workspace-full

RUN sudo apt update -y
RUN sudo apt install -y python-opengl
RUN sudo apt install -y xvfb
