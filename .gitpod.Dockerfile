FROM gitpod/workspace-full-vnc

USER gitpod

RUN sudo apt update -y
RUN sudo apt install -y \
  python-opengl \
  xvfb \
  libgtk-3-dev \
  libx11-dev libxkbfile-dev \
  libsecret-1-dev \
  libgconf2–4 \
  libnss3
