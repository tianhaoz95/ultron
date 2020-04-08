FROM gitpod/workspace-full-vnc

USER gitpod

RUN sudo apt update -y
RUN sudo apt install -y \
  python-opengl \
  xvfb
